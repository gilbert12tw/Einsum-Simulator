#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cutensor.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

#define CHECK_CUDA(x)                                                     \
    do {                                                                  \
        cudaError_t _err = (x);                                           \
        if (_err != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +        \
                                     cudaGetErrorString(_err));           \
        }                                                                 \
    } while (0)

#define CHECK_CUTENSOR(x)                                                 \
    do {                                                                  \
        cutensorStatus_t _err = (x);                                      \
        if (_err != CUTENSOR_STATUS_SUCCESS) {                            \
            throw std::runtime_error(std::string("cuTENSOR error: ") +    \
                                     cutensorGetErrorString(_err));       \
        }                                                                 \
    } while (0)

namespace {

std::vector<std::string> split_utf8_string(const std::string &str) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < str.length();) {
        const unsigned char c = static_cast<unsigned char>(str[i]);
        size_t char_len = 1;
        if (c < 0x80) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }
        if (i + char_len <= str.length()) {
            chars.push_back(str.substr(i, char_len));
        }
        i += char_len;
    }
    return chars;
}

std::vector<std::string> vec_difference(const std::vector<std::string> &a, const std::vector<std::string> &b) {
    std::vector<std::string> out;
    for (const auto &item : a) {
        if (std::find(b.begin(), b.end(), item) == b.end()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<std::string> vec_union_preserve(const std::vector<std::string> &a, const std::vector<std::string> &b) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto &item : a) {
        out.push_back(item);
        seen.insert(item);
    }
    for (const auto &item : b) {
        if (seen.find(item) == seen.end()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<std::string> contraction_modes(const std::vector<std::string> &left, const std::vector<std::string> &right) {
    auto l = vec_difference(left, right);
    auto r = vec_difference(right, left);
    return vec_union_preserve(l, r);
}

std::vector<std::string> split_by_comma(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::vector<int> parse_all_ints(const std::string &s) {
    std::vector<int> out;
    int sign = 1;
    long val = 0;
    bool in_num = false;
    for (char c : s) {
        if (c == '-') {
            if (in_num) {
                out.push_back(static_cast<int>(sign * val));
                val = 0;
            }
            sign = -1;
            in_num = true;
        } else if (c >= '0' && c <= '9') {
            if (!in_num) {
                in_num = true;
                sign = 1;
                val = 0;
            }
            val = val * 10 + (c - '0');
        } else if (in_num) {
            out.push_back(static_cast<int>(sign * val));
            in_num = false;
            sign = 1;
            val = 0;
        }
    }
    if (in_num) {
        out.push_back(static_cast<int>(sign * val));
    }
    return out;
}

inline uint64_t triple_key(int a, int b, int c) {
    return (static_cast<uint64_t>(a) << 42) |
           (static_cast<uint64_t>(b) << 21) |
           static_cast<uint64_t>(c);
}

std::string get_config_path() {
    const char *env = std::getenv("QSVM_TENSOR_CONFIG");
    if (env && env[0] != '\0') {
        return std::string(env);
    }
    throw std::runtime_error(
        "QSVM_TENSOR_CONFIG is not set. Prefer calling init_backend(config) from Python.");
}

struct BackendContext {
    // Static contraction metadata. Initial tensors occupy [0, initial_tensor_count);
    // intermediate tensor IDs follow the flattened contraction triples.
    int n_steps = 0;
    int n_tensors = 0;
    int initial_tensor_count = 0;
    std::vector<int> triple_path;  // flattened triples
    // Optional three-stream schedule. If all three paths are provided, stream0
    // and stream1 run independent subtrees before stream2 joins them at the
    // synchronization contraction.
    std::vector<int> stream_0_path;
    std::vector<int> stream_1_path;
    std::vector<int> stream_2_path;
    std::unordered_map<uint64_t, int> triple_to_step;
    bool use_multistream = false;
    std::vector<std::vector<int>> tensor_modes;
    std::vector<std::vector<int64_t>> tensor_extents;
    std::vector<std::vector<int64_t>> tensor_strides;
    std::vector<size_t> tensor_elements;
    std::vector<size_t> tensor_bytes;
    // Reused device buffers for intermediate tensors. Input tensor pointers can
    // be overwritten with torch-owned CUDA pointers for zero-copy execution.
    std::vector<void *> device_tensors;

    cutensorHandle_t handle{};
    std::vector<cutensorTensorDescriptor_t> tensor_descs;
    std::vector<cutensorOperationDescriptor_t> op_descs;
    std::vector<cutensorPlanPreference_t> plan_prefs;
    std::vector<cutensorPlan_t> plans;
    std::vector<uint64_t> workspace_sizes;
    uint64_t max_workspace_size = 0;
    // cuTENSOR workspace must not be shared by concurrently executing streams.
    void *workspace_single = nullptr;
    void *workspace_stream0 = nullptr;
    void *workspace_stream1 = nullptr;
    void *workspace_stream2 = nullptr;
    cudaStream_t stream0{};
    cudaStream_t stream1{};
    cudaStream_t stream2{};
    cudaEvent_t stream0_done{};
    cudaEvent_t stream1_done{};
    cudaEvent_t sample_done{};

    BackendContext() {
        auto config_path = get_config_path();
        load_config_from_file(config_path);
        build_modes_and_shapes();
        init_cutensor();
    }

    BackendContext(
        const std::vector<std::string> &input_subscripts,
        const std::vector<int> &flat_triple_path,
        const std::vector<int> &flat_stream_0_path,
        const std::vector<int> &flat_stream_1_path,
        const std::vector<int> &flat_stream_2_path) {
        load_config_from_data(
            input_subscripts,
            flat_triple_path,
            flat_stream_0_path,
            flat_stream_1_path,
            flat_stream_2_path);
        build_modes_and_shapes();
        init_cutensor();
    }

    ~BackendContext() {
        for (void *ptr : device_tensors) {
            if (ptr) cudaFree(ptr);
        }
        if (workspace_single) cudaFree(workspace_single);
        if (workspace_stream0) cudaFree(workspace_stream0);
        if (workspace_stream1) cudaFree(workspace_stream1);
        if (workspace_stream2) cudaFree(workspace_stream2);
        for (auto p : plans) {
            cutensorDestroyPlan(p);
        }
        for (auto p : plan_prefs) {
            cutensorDestroyPlanPreference(p);
        }
        for (auto o : op_descs) {
            cutensorDestroyOperationDescriptor(o);
        }
        for (auto t : tensor_descs) {
            cutensorDestroyTensorDescriptor(t);
        }
        if (stream0_done) cudaEventDestroy(stream0_done);
        if (stream1_done) cudaEventDestroy(stream1_done);
        if (sample_done) cudaEventDestroy(sample_done);
        if (stream0) cudaStreamDestroy(stream0);
        if (stream1) cudaStreamDestroy(stream1);
        if (stream2) cudaStreamDestroy(stream2);
        if (handle) {
            cutensorDestroy(handle);
        }
    }

    void load_config_from_file(const std::string &config_path) {
        std::ifstream f(config_path);
        if (!f) {
            throw std::runtime_error("Failed to open tensor config: " + config_path);
        }
        std::string line1, line2, line3, line4, line5;
        std::getline(f, line1);
        std::getline(f, line2);
        std::getline(f, line3);
        std::getline(f, line4);
        std::getline(f, line5);
        if (line1.empty() || line5.empty()) {
            throw std::runtime_error("Invalid tensor config format: " + config_path);
        }
        load_config_from_data(
            split_by_comma(line1),
            parse_all_ints(line5),
            parse_all_ints(line2),
            parse_all_ints(line3),
            parse_all_ints(line4));
    }

    void load_config_from_data(
        const std::vector<std::string> &input_subscripts,
        const std::vector<int> &flat_triple_path,
        const std::vector<int> &flat_stream_0_path = {},
        const std::vector<int> &flat_stream_1_path = {},
        const std::vector<int> &flat_stream_2_path = {}) {
        initial_tensor_count = static_cast<int>(input_subscripts.size());
        triple_path = flat_triple_path;
        stream_0_path = flat_stream_0_path;
        stream_1_path = flat_stream_1_path;
        stream_2_path = flat_stream_2_path;
        if (triple_path.empty() || triple_path.size() % 3 != 0) {
            throw std::runtime_error("Invalid triple path in backend init config.");
        }
        if ((!stream_0_path.empty() && stream_0_path.size() % 3 != 0) ||
            (!stream_1_path.empty() && stream_1_path.size() % 3 != 0) ||
            (!stream_2_path.empty() && stream_2_path.size() % 3 != 0)) {
            throw std::runtime_error("Invalid stream path format in backend init config.");
        }
        n_steps = static_cast<int>(triple_path.size() / 3);
        n_tensors = triple_path.back() + 1;
        use_multistream = !stream_0_path.empty() && !stream_1_path.empty() && !stream_2_path.empty();

        // Temporarily store initial subscripts in tensor_modes placeholder via map later.
        tensor_modes.assign(n_tensors, {});

        // Build edge map.
        std::set<std::string> all_edges;
        for (const auto &sub : input_subscripts) {
            auto chars = split_utf8_string(sub);
            for (const auto &ch : chars) {
                all_edges.insert(ch);
            }
        }
        std::unordered_map<std::string, int> edge_to_int;
        int edge_id = 0;
        for (const auto &e : all_edges) {
            edge_to_int[e] = edge_id++;
        }

        // Build tensor symbolic list.
        std::vector<std::vector<std::string>> tensor_list(n_tensors);
        for (int i = 0; i < initial_tensor_count; ++i) {
            tensor_list[i] = split_utf8_string(input_subscripts[i]);
        }
        triple_to_step.clear();
        triple_to_step.reserve(static_cast<size_t>(n_steps) * 2);
        for (int step = 0; step < n_steps; ++step) {
            int a = triple_path[step * 3 + 0];
            int b = triple_path[step * 3 + 1];
            int c = triple_path[step * 3 + 2];
            if (a < 0 || b < 0 || c < 0 || a >= n_tensors || b >= n_tensors || c >= n_tensors) {
                throw std::runtime_error("Out-of-range tensor id in contraction path");
            }
            triple_to_step[triple_key(a, b, c)] = step;
            tensor_list[c] = contraction_modes(tensor_list[a], tensor_list[b]);
        }

        auto validate_stream_path = [&](const std::vector<int> &flat_path, const char *name) {
            for (size_t i = 0; i < flat_path.size(); i += 3) {
                const int a = flat_path[i + 0];
                const int b = flat_path[i + 1];
                const int c = flat_path[i + 2];
                if (triple_to_step.find(triple_key(a, b, c)) == triple_to_step.end()) {
                    throw std::runtime_error(std::string("Unknown contraction in ") + name);
                }
            }
        };
        validate_stream_path(stream_0_path, "stream_0");
        validate_stream_path(stream_1_path, "stream_1");
        validate_stream_path(stream_2_path, "stream_2");

        for (int i = 0; i < n_tensors; ++i) {
            std::vector<int> modes;
            modes.reserve(tensor_list[i].size());
            for (const auto &e : tensor_list[i]) {
                auto it = edge_to_int.find(e);
                if (it == edge_to_int.end()) {
                    throw std::runtime_error("Missing edge id mapping.");
                }
                modes.push_back(it->second);
            }
            tensor_modes[i] = std::move(modes);
        }
    }

    void build_modes_and_shapes() {
        tensor_extents.resize(n_tensors);
        tensor_strides.resize(n_tensors);
        tensor_elements.resize(n_tensors, 1);
        tensor_bytes.resize(n_tensors, 0);

        for (int i = 0; i < n_tensors; ++i) {
            auto dims = static_cast<int>(tensor_modes[i].size());
            std::vector<int64_t> ext(dims, 2);
            std::vector<int64_t> str(dims, 1);
            for (int k = dims - 2; k >= 0; --k) {
                str[k] = str[k + 1] * ext[k + 1];
            }
            tensor_extents[i] = std::move(ext);
            tensor_strides[i] = std::move(str);

            size_t elems = 1;
            for (int k = 0; k < dims; ++k) {
                elems *= 2;
            }
            tensor_elements[i] = elems;
            tensor_bytes[i] = elems * sizeof(cuDoubleComplex);
        }
    }

    void init_cutensor() {
        CHECK_CUTENSOR(cutensorCreate(&handle));
        CHECK_CUDA(cudaStreamCreate(&stream0));
        if (use_multistream) {
            CHECK_CUDA(cudaStreamCreate(&stream1));
            CHECK_CUDA(cudaStreamCreate(&stream2));
            CHECK_CUDA(cudaEventCreateWithFlags(&stream0_done, cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&stream1_done, cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&sample_done, cudaEventDisableTiming));

        const uint32_t alignment = 128;
        tensor_descs.resize(n_tensors);
        for (int i = 0; i < n_tensors; ++i) {
            CHECK_CUTENSOR(cutensorCreateTensorDescriptor(
                handle,
                &tensor_descs[i],
                static_cast<int32_t>(tensor_modes[i].size()),
                tensor_extents[i].data(),
                tensor_strides[i].empty() ? nullptr : tensor_strides[i].data(),
                CUTENSOR_C_64F,
                alignment));
        }

        op_descs.resize(n_steps);
        plan_prefs.resize(n_steps);
        plans.resize(n_steps);
        workspace_sizes.resize(n_steps, 0);

        for (int step = 0; step < n_steps; ++step) {
            const int a = triple_path[step * 3 + 0];
            const int b = triple_path[step * 3 + 1];
            const int c = triple_path[step * 3 + 2];
            CHECK_CUTENSOR(cutensorCreatePlanPreference(
                handle,
                &plan_prefs[step],
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE));

            CHECK_CUTENSOR(cutensorCreateContraction(
                handle,
                &op_descs[step],
                tensor_descs[a], tensor_modes[a].data(), CUTENSOR_OP_IDENTITY,
                tensor_descs[b], tensor_modes[b].data(), CUTENSOR_OP_IDENTITY,
                tensor_descs[c], tensor_modes[c].data(), CUTENSOR_OP_IDENTITY,
                tensor_descs[c], tensor_modes[c].data(),
                CUTENSOR_COMPUTE_DESC_64F));

            uint64_t ws_est = 0;
            CHECK_CUTENSOR(cutensorEstimateWorkspaceSize(
                handle,
                op_descs[step],
                plan_prefs[step],
                CUTENSOR_WORKSPACE_DEFAULT,
                &ws_est));

            CHECK_CUTENSOR(cutensorCreatePlan(
                handle,
                &plans[step],
                op_descs[step],
                plan_prefs[step],
                ws_est));

            CHECK_CUTENSOR(cutensorPlanGetAttribute(
                handle,
                plans[step],
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &workspace_sizes[step],
                sizeof(workspace_sizes[step])));

            max_workspace_size = std::max(max_workspace_size, workspace_sizes[step]);
        }

        if (max_workspace_size > 0) {
            if (use_multistream) {
                CHECK_CUDA(cudaMalloc(&workspace_stream0, max_workspace_size));
                CHECK_CUDA(cudaMalloc(&workspace_stream1, max_workspace_size));
                CHECK_CUDA(cudaMalloc(&workspace_stream2, max_workspace_size));
            } else {
                CHECK_CUDA(cudaMalloc(&workspace_single, max_workspace_size));
            }
        }

        device_tensors.assign(n_tensors, nullptr);
        for (int i = 0; i < n_tensors; ++i) {
            CHECK_CUDA(cudaMalloc(&device_tensors[i], tensor_bytes[i]));
        }
    }

    void run_one_step(
        int step,
        const std::vector<void *> &d_tensors,
        cudaStream_t exec_stream,
        void *workspace_ptr) const {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        const int a = triple_path[step * 3 + 0];
        const int b = triple_path[step * 3 + 1];
        const int c = triple_path[step * 3 + 2];
        CHECK_CUTENSOR(cutensorContract(
            handle,
            plans[step],
            &alpha,
            d_tensors[a],
            d_tensors[b],
            &beta,
            d_tensors[c],
            d_tensors[c],
            workspace_ptr,
            workspace_sizes[step],
            exec_stream));
    }

    void run_flat_path(
        const std::vector<int> &flat_path,
        const std::vector<void *> &d_tensors,
        cudaStream_t exec_stream,
        void *workspace_ptr) const {
        for (size_t i = 0; i < flat_path.size(); i += 3) {
            const int a = flat_path[i + 0];
            const int b = flat_path[i + 1];
            const int c = flat_path[i + 2];
            const auto it = triple_to_step.find(triple_key(a, b, c));
            if (it == triple_to_step.end()) {
                throw std::runtime_error("Missing contraction step for stream path item.");
            }
            run_one_step(it->second, d_tensors, exec_stream, workspace_ptr);
        }
    }

    void contract_with_buffers(const std::vector<void *> &d_tensors) const {
        // Single-stream mode replays the full path in optimizer order.
        if (!use_multistream) {
            for (int step = 0; step < n_steps; ++step) {
                run_one_step(step, d_tensors, stream0, workspace_single);
            }
            return;
        }

        // Multi-stream mode: two independent subtrees run in parallel, then
        // stream2 waits on both events and completes the root-side contractions.
        run_flat_path(stream_0_path, d_tensors, stream0, workspace_stream0);
        run_flat_path(stream_1_path, d_tensors, stream1, workspace_stream1);
        CHECK_CUDA(cudaEventRecord(stream0_done, stream0));
        CHECK_CUDA(cudaEventRecord(stream1_done, stream1));
        CHECK_CUDA(cudaStreamWaitEvent(stream2, stream0_done, 0));
        CHECK_CUDA(cudaStreamWaitEvent(stream2, stream1_done, 0));
        run_flat_path(stream_2_path, d_tensors, stream2, workspace_stream2);
    }
};

std::mutex g_ctx_mutex;
std::unique_ptr<BackendContext> g_ctx;

std::vector<int> parse_py_triple_path(const py::object &obj) {
    py::sequence seq = py::cast<py::sequence>(obj);
    std::vector<int> out;
    if (py::len(seq) == 0) {
        return out;
    }
    py::object first = seq[0];
    if (py::isinstance<py::int_>(first)) {
        out.reserve(static_cast<size_t>(py::len(seq)));
        for (auto item : seq) {
            out.push_back(py::cast<int>(item));
        }
        return out;
    }
    out.reserve(static_cast<size_t>(py::len(seq)) * 3);
    for (auto item : seq) {
        py::sequence triple = py::cast<py::sequence>(item);
        if (py::len(triple) != 3) {
            throw std::runtime_error("Each contraction step must be a triple [a,b,c].");
        }
        out.push_back(py::cast<int>(triple[0]));
        out.push_back(py::cast<int>(triple[1]));
        out.push_back(py::cast<int>(triple[2]));
    }
    return out;
}

void init_backend_from_dict(const py::dict &config) {
    if (!config.contains("input_subscripts")) {
        throw std::runtime_error("init_backend(config): missing key 'input_subscripts'");
    }
    if (!config.contains("triple_path")) {
        throw std::runtime_error("init_backend(config): missing key 'triple_path'");
    }
    std::vector<std::string> input_subscripts = py::cast<std::vector<std::string>>(config["input_subscripts"]);
    std::vector<int> flat_triple_path = parse_py_triple_path(py::cast<py::object>(config["triple_path"]));
    std::vector<int> flat_stream_0_path;
    std::vector<int> flat_stream_1_path;
    std::vector<int> flat_stream_2_path;
    if (config.contains("stream_0")) {
        flat_stream_0_path = parse_py_triple_path(py::cast<py::object>(config["stream_0"]));
    }
    if (config.contains("stream_1")) {
        flat_stream_1_path = parse_py_triple_path(py::cast<py::object>(config["stream_1"]));
    }
    if (config.contains("stream_2")) {
        flat_stream_2_path = parse_py_triple_path(py::cast<py::object>(config["stream_2"]));
    }
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    g_ctx.reset(new BackendContext(
        input_subscripts,
        flat_triple_path,
        flat_stream_0_path,
        flat_stream_1_path,
        flat_stream_2_path));
}

BackendContext &get_context() {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    if (!g_ctx) {
        g_ctx.reset(new BackendContext());
    }
    return *g_ctx;
}

}  // namespace

static py::array_t<double> contract_batch_from_numpy(const py::list &opers_np) {
    auto &ctx = get_context();
    const auto batch = static_cast<ssize_t>(py::len(opers_np));
    py::array_t<double> out(batch);
    auto r = out.mutable_unchecked<1>();
    auto &d_tensors = ctx.device_tensors;
    std::vector<const void *> prev_host_ptr(ctx.initial_tensor_count, nullptr);

    std::vector<cuDoubleComplex> final_vals(static_cast<size_t>(batch), make_cuDoubleComplex(0.0, 0.0));
    cudaStream_t result_stream = ctx.use_multistream ? ctx.stream2 : ctx.stream0;
    for (ssize_t i = 0; i < batch; ++i) {
        if (i > 0) {
            CHECK_CUDA(cudaStreamWaitEvent(ctx.stream0, ctx.sample_done, 0));
            if (ctx.use_multistream) {
                CHECK_CUDA(cudaStreamWaitEvent(ctx.stream1, ctx.sample_done, 0));
            }
        }
        py::list sample = py::cast<py::list>(opers_np[i]);
        if (static_cast<int>(py::len(sample)) < ctx.initial_tensor_count) {
            throw std::runtime_error("Sample operand count is smaller than initial tensor count.");
        }

        // Copy initial tensors from host to device.
        for (int t = 0; t < ctx.initial_tensor_count; ++t) {
            py::array arr = py::cast<py::array>(sample[t]);
            py::array arr_c = py::array::ensure(arr);
            if (!arr_c) {
                throw std::runtime_error("Invalid numpy array in operands.");
            }
            if (!py::isinstance<py::array_t<std::complex<double>>>(arr_c)) {
                throw std::runtime_error("All operand arrays must be complex128.");
            }
            auto buf = arr_c.request();
            const size_t expected = ctx.tensor_elements[t];
            if (static_cast<size_t>(buf.size) != expected) {
                throw std::runtime_error("Operand size mismatch at tensor " + std::to_string(t));
            }
            if (i == 0 || prev_host_ptr[t] != buf.ptr) {
                CHECK_CUDA(cudaMemcpy(
                    d_tensors[t],
                    buf.ptr,
                    ctx.tensor_bytes[t],
                    cudaMemcpyHostToDevice));
                prev_host_ptr[t] = buf.ptr;
            }
        }

        ctx.contract_with_buffers(d_tensors);

        CHECK_CUDA(cudaMemcpyAsync(
            &final_vals[static_cast<size_t>(i)],
            d_tensors[ctx.n_tensors - 1],
            sizeof(cuDoubleComplex),
            cudaMemcpyDeviceToHost,
            result_stream));
        CHECK_CUDA(cudaEventRecord(ctx.sample_done, result_stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(result_stream));
    for (ssize_t i = 0; i < batch; ++i) {
        const cuDoubleComplex v = final_vals[static_cast<size_t>(i)];
        const double re = cuCreal(v);
        const double im = cuCimag(v);
        r(i) = re * re + im * im;
    }
    return out;
}

static py::array_t<double> contract_batch_from_torch(const py::object &opers_torch) {
    auto &ctx = get_context();
    py::list opers = py::cast<py::list>(opers_torch);
    const auto batch = static_cast<ssize_t>(py::len(opers));
    py::array_t<double> out(batch);
    auto r = out.mutable_unchecked<1>();

    std::vector<cuDoubleComplex> final_vals(static_cast<size_t>(batch), make_cuDoubleComplex(0.0, 0.0));
    cudaStream_t result_stream = ctx.use_multistream ? ctx.stream2 : ctx.stream0;
    for (ssize_t i = 0; i < batch; ++i) {
        if (i > 0) {
            CHECK_CUDA(cudaStreamWaitEvent(ctx.stream0, ctx.sample_done, 0));
            if (ctx.use_multistream) {
                CHECK_CUDA(cudaStreamWaitEvent(ctx.stream1, ctx.sample_done, 0));
            }
        }
        py::list sample = py::cast<py::list>(opers[i]);
        if (static_cast<int>(py::len(sample)) < ctx.initial_tensor_count) {
            throw std::runtime_error("Sample operand count is smaller than initial tensor count.");
        }

        // Start from preallocated intermediate buffers.
        std::vector<void *> d_tensors = ctx.device_tensors;

        // Use torch CUDA tensor device pointers directly for initial tensors.
        for (int t = 0; t < ctx.initial_tensor_count; ++t) {
            py::object ten = py::cast<py::object>(sample[t]);
            if (i == 0) {
                bool is_cuda = py::cast<bool>(ten.attr("is_cuda"));
                if (!is_cuda) {
                    throw std::runtime_error("contract_batch_from_torch requires CUDA tensors.");
                }
                bool is_contig = py::cast<bool>(ten.attr("is_contiguous")());
                if (!is_contig) {
                    throw std::runtime_error("Torch operand tensor must be contiguous.");
                }
                auto dtype_str = py::cast<std::string>(py::str(ten.attr("dtype")));
                if (dtype_str != "torch.complex128") {
                    throw std::runtime_error("Torch operand tensor dtype must be torch.complex128.");
                }
                int64_t numel = py::cast<int64_t>(ten.attr("numel")());
                if (static_cast<size_t>(numel) != ctx.tensor_elements[t]) {
                    throw std::runtime_error("Torch operand numel mismatch at tensor " + std::to_string(t));
                }
            }
            uint64_t ptr_u64 = py::cast<uint64_t>(ten.attr("data_ptr")());
            d_tensors[t] = reinterpret_cast<void *>(ptr_u64);
        }

        ctx.contract_with_buffers(d_tensors);

        CHECK_CUDA(cudaMemcpyAsync(
            &final_vals[static_cast<size_t>(i)],
            d_tensors[ctx.n_tensors - 1],
            sizeof(cuDoubleComplex),
            cudaMemcpyDeviceToHost,
            result_stream));
        CHECK_CUDA(cudaEventRecord(ctx.sample_done, result_stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(result_stream));
    for (ssize_t i = 0; i < batch; ++i) {
        const cuDoubleComplex v = final_vals[static_cast<size_t>(i)];
        const double re = cuCreal(v);
        const double im = cuCimag(v);
        r(i) = re * re + im * im;
    }
    return out;
}

static py::array_t<double> contract_batch_from_torch_ptr_table(
    const py::array_t<uint64_t, py::array::c_style | py::array::forcecast> &ptr_table) {
    auto &ctx = get_context();
    auto ptrs = ptr_table.unchecked<2>();
    const ssize_t batch = ptrs.shape(0);
    const ssize_t n_cols = ptrs.shape(1);
    if (n_cols != static_cast<ssize_t>(ctx.initial_tensor_count)) {
        throw std::runtime_error("ptr_table second dimension must equal initial_tensor_count.");
    }

    py::array_t<double> out(batch);
    auto r = out.mutable_unchecked<1>();
    std::vector<cuDoubleComplex> final_vals(static_cast<size_t>(batch), make_cuDoubleComplex(0.0, 0.0));

    cuDoubleComplex *d_results = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_results), static_cast<size_t>(batch) * sizeof(cuDoubleComplex)));
    try {
        auto &d_tensors = ctx.device_tensors;
        cudaStream_t result_stream = ctx.use_multistream ? ctx.stream2 : ctx.stream0;
        for (ssize_t i = 0; i < batch; ++i) {
            if (i > 0) {
                CHECK_CUDA(cudaStreamWaitEvent(ctx.stream0, ctx.sample_done, 0));
                if (ctx.use_multistream) {
                    CHECK_CUDA(cudaStreamWaitEvent(ctx.stream1, ctx.sample_done, 0));
                }
            }
            for (int t = 0; t < ctx.initial_tensor_count; ++t) {
                const uint64_t ptr_u64 = ptrs(i, t);
                if (ptr_u64 == 0) {
                    throw std::runtime_error("ptr_table contains null pointer.");
                }
                d_tensors[t] = reinterpret_cast<void *>(ptr_u64);
            }

            ctx.contract_with_buffers(d_tensors);

            CHECK_CUDA(cudaMemcpyAsync(
                d_results + i,
                d_tensors[ctx.n_tensors - 1],
                sizeof(cuDoubleComplex),
                cudaMemcpyDeviceToDevice,
                result_stream));
            CHECK_CUDA(cudaEventRecord(ctx.sample_done, result_stream));
        }

        CHECK_CUDA(cudaStreamSynchronize(result_stream));
        CHECK_CUDA(cudaMemcpy(
            final_vals.data(),
            d_results,
            static_cast<size_t>(batch) * sizeof(cuDoubleComplex),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_results));
    } catch (...) {
        if (d_results) {
            cudaFree(d_results);
        }
        throw;
    }

    for (ssize_t i = 0; i < batch; ++i) {
        const cuDoubleComplex v = final_vals[static_cast<size_t>(i)];
        const double re = cuCreal(v);
        const double im = cuCimag(v);
        r(i) = re * re + im * im;
    }
    return out;
}

static py::array_t<std::complex<double>> contract_batch_complex_from_torch_ptr_table(
    const py::array_t<uint64_t, py::array::c_style | py::array::forcecast> &ptr_table) {
    auto &ctx = get_context();
    auto ptrs = ptr_table.unchecked<2>();
    const ssize_t batch = ptrs.shape(0);
    const ssize_t n_cols = ptrs.shape(1);
    if (n_cols != static_cast<ssize_t>(ctx.initial_tensor_count)) {
        throw std::runtime_error("ptr_table second dimension must equal initial_tensor_count.");
    }

    py::array_t<std::complex<double>> out(batch);
    auto r = out.mutable_unchecked<1>();
    std::vector<cuDoubleComplex> final_vals(static_cast<size_t>(batch), make_cuDoubleComplex(0.0, 0.0));

    // Store per-sample scalar outputs on device first, then copy them back as
    // one contiguous block. This avoids many small D2H copies.
    cuDoubleComplex *d_results = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_results), static_cast<size_t>(batch) * sizeof(cuDoubleComplex)));
    try {
        auto &d_tensors = ctx.device_tensors;
        cudaStream_t result_stream = ctx.use_multistream ? ctx.stream2 : ctx.stream0;
        for (ssize_t i = 0; i < batch; ++i) {
            if (i > 0) {
                CHECK_CUDA(cudaStreamWaitEvent(ctx.stream0, ctx.sample_done, 0));
                if (ctx.use_multistream) {
                    CHECK_CUDA(cudaStreamWaitEvent(ctx.stream1, ctx.sample_done, 0));
                }
            }
            for (int t = 0; t < ctx.initial_tensor_count; ++t) {
                const uint64_t ptr_u64 = ptrs(i, t);
                if (ptr_u64 == 0) {
                    throw std::runtime_error("ptr_table contains null pointer.");
                }
                d_tensors[t] = reinterpret_cast<void *>(ptr_u64);
            }

            ctx.contract_with_buffers(d_tensors);

            CHECK_CUDA(cudaMemcpyAsync(
                d_results + i,
                d_tensors[ctx.n_tensors - 1],
                sizeof(cuDoubleComplex),
                cudaMemcpyDeviceToDevice,
                result_stream));
            CHECK_CUDA(cudaEventRecord(ctx.sample_done, result_stream));
        }

        CHECK_CUDA(cudaStreamSynchronize(result_stream));
        CHECK_CUDA(cudaMemcpy(
            final_vals.data(),
            d_results,
            static_cast<size_t>(batch) * sizeof(cuDoubleComplex),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_results));
    } catch (...) {
        if (d_results) {
            cudaFree(d_results);
        }
        throw;
    }

    for (ssize_t i = 0; i < batch; ++i) {
        const cuDoubleComplex v = final_vals[static_cast<size_t>(i)];
        r(i) = std::complex<double>(cuCreal(v), cuCimag(v));
    }
    return out;
}

PYBIND11_MODULE(qsvm_cutensor_backend, m) {
    m.doc() = "QSVM cuTENSOR backend module";
    m.def("init_backend", &init_backend_from_dict, "Initialize backend with in-memory contraction config");
    m.def("contract_batch_from_numpy", &contract_batch_from_numpy, "Contract batch from numpy operands");
    m.def("contract_batch_from_torch", &contract_batch_from_torch, "Contract batch from torch operands");
    m.def("contract_batch_from_torch_ptr_table", &contract_batch_from_torch_ptr_table, "Contract batch from torch pointer table");
    m.def("contract_batch_complex_from_torch_ptr_table", &contract_batch_complex_from_torch_ptr_table, "Contract batch from torch pointer table and return complex amplitudes");
    m.def("version", []() { return std::string("0.10.0-cutensor"); });
}

