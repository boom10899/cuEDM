#include <iostream>

#include <argh.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#include "cross_mapping_cpu.h"
#include "data_frame.h"
#include "embedding_dim_cpu.h"
#include "mpi_master.h"
#include "mpi_worker.h"
#ifdef ENABLE_GPU_KERNEL
#include "cross_mapping_gpu.h"
#include "embedding_dim_gpu.h"
#endif
#include "stats.h"
#include "timer.h"

struct Parameters {
    std::string input_fname;
    std::string output_fname;
    uint32_t tau;
    uint32_t Tp;
    uint32_t max_E;
    std::string kernel_type;
    std::string dataset_name;
    uint32_t chunk_size;
    bool verbose;
};

class EmbeddingDimMPIMaster : public MPIMaster
{
public:
    std::vector<uint32_t> optimal_E;

    EmbeddingDimMPIMaster(const DataFrame df, MPI_Comm comm)
        : MPIMaster(comm), optimal_E(df.n_columns()), current_id(0),
          dataframe(df)
    {
    }
    ~EmbeddingDimMPIMaster() {}

protected:
    uint32_t current_id;
    DataFrame dataframe;

    void next_task(nlohmann::json &task) override
    {
        task["id"] = current_id;
        current_id++;
    }

    bool task_left() const override
    {
        return current_id < dataframe.n_columns();
    }

    void task_done(const nlohmann::json &result) override
    {
        std::cout << "Timeseries #" << result["id"] << " best E=" << result["E"]
                  << std::endl;

        optimal_E[result["id"]] = result["E"];
    }
};

template <class T> class EmbeddingDimMPIWorker : public MPIWorker
{
public:
    EmbeddingDimMPIWorker(const DataFrame df, bool verbose, MPI_Comm comm)
        : MPIWorker(comm),
          embedding_dim(std::unique_ptr<EmbeddingDim>(new T(20, 1, 1, true))),
          dataframe(df), verbose(verbose)
    {
    }
    ~EmbeddingDimMPIWorker() {}

    double total_knn_time() { return timer_knn; }
    double total_lookup_time() { return timer_lookup; }
    double total_cpu_to_gpu_time() { return timer_cpu_to_gpu; }
    double total_gpu_to_cpu_time() { return timer_gpu_to_cpu; }

protected:
    std::unique_ptr<EmbeddingDim> embedding_dim;
    DataFrame dataframe;
    bool verbose;
    float timer_knn = 0;
    float timer_lookup = 0;
    float timer_cpu_to_gpu = 0;
    float timer_gpu_to_cpu = 0;

    void do_task(nlohmann::json &result, const nlohmann::json &task) override
    {
        const auto id = task["id"];
        const auto ts = dataframe.columns[id];
        const auto best_E =
            embedding_dim->run(ts);

        timer_knn = embedding_dim->get_timer_knn_elapsed();
        timer_lookup = embedding_dim->get_timer_lookup_elapsed();
        timer_cpu_to_gpu = embedding_dim->get_timer_cpu_to_gpu_elapsed();
        timer_gpu_to_cpu = embedding_dim->get_timer_gpu_to_cpu_elapsed();

        result["id"] = id;
        result["E"] = best_E;

        // std::cout << id << " | " << timer_knn << " | " << timer_lookup << " | " << timer_cpu_to_gpu << " | " << timer_cpu_to_gpu << std::endl;
    }
};

class CrossMappingMPIMaster : public MPIMaster
{
public:
    CrossMappingMPIMaster(const DataFrame df, uint32_t chunk_size,
                          MPI_Comm comm)
        : MPIMaster(comm), current_id(0), dataframe(df), chunk_size(chunk_size)
    {
    }
    ~CrossMappingMPIMaster() {}

protected:
    size_t current_id;
    DataFrame dataframe;
    size_t chunk_size;

    void next_task(nlohmann::json &task) override
    {
        task["start_id"] = current_id;
        task["stop_id"] =
            std::min(current_id + chunk_size, dataframe.n_columns());
        current_id += chunk_size;
    }

    bool task_left() const override
    {
        return current_id < dataframe.n_columns();
    }

    void task_done(const nlohmann::json &result) override
    {
        std::cout << "Timeseries #" << result["start_id"] << " - #"
                  << result["stop_id"].get<int>() - 1 << " finished."
                  << std::endl;
    }
};

template <class T> class CrossMappingMPIWorker : public MPIWorker
{
public:
    CrossMappingMPIWorker(HighFive::DataSet dataset, const DataFrame df,
                          const std::vector<uint32_t> optimal_E, bool verbose,
                          MPI_Comm comm)
        : MPIWorker(comm), dataset(dataset),
          xmap(std::unique_ptr<CrossMapping>(new T(20, 1, 0, true))),
          dataframe(df), optimal_E(optimal_E), verbose(verbose)
    {
    }
    ~CrossMappingMPIWorker() {}

    float total_io_time() { return timer_io.elapsed(); }
    float total_knn_time() { return timer_knn.elapsed(); }
    float total_lookup_time() { return timer_lookup.elapsed(); }
    float total_cpu_to_gpu_time() { return timer_cpu_to_gpu; }
    float total_gpu_to_cpu_time() { return timer_gpu_to_cpu; }

protected:
    HighFive::DataSet dataset;
    std::unique_ptr<CrossMapping> xmap;
    DataFrame dataframe;
    std::vector<uint32_t> optimal_E;
    bool verbose;
    Timer timer_io;
    Timer timer_knn;
    Timer timer_lookup;
    float timer_cpu_to_gpu = 0;
    float timer_gpu_to_cpu = 0;

    void do_task(nlohmann::json &result, const nlohmann::json &task) override
    {
        const uint32_t start_id = task["start_id"];
        const uint32_t stop_id = task["stop_id"];
        uint32_t task_size = stop_id - start_id;

        std::vector<std::vector<float>> rhos(
            task_size, std::vector<float>(dataframe.n_columns()));

        for (uint32_t i = 0; i < task_size; i++) {
            const auto library = dataframe.columns[start_id + i];
            xmap->run(rhos[i], library, dataframe.columns, optimal_E,
                      timer_knn, timer_lookup);
        }

        timer_cpu_to_gpu = xmap->get_timer_cpu_to_gpu_elapsed();
        timer_gpu_to_cpu = xmap->get_timer_gpu_to_cpu_elapsed();

        timer_io.start();
        dataset.select({start_id, 0}, {task_size, dataframe.n_columns()})
            .write(rhos);
        timer_io.stop();

        result["start_id"] = start_id;
        result["stop_id"] = stop_id;

        // std::cout << start_id << " | " << timer_knn.elapsed() << " | " << timer_lookup.elapsed() << " | " << timer_cpu_to_gpu << " | " << timer_cpu_to_gpu << std::endl;

    }
};

bool ends_with(const std::string &str, const std::string &suffix)
{
    if (str.size() < suffix.size()) {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void run(int rank, const DataFrame &df, const Parameters &parameters, Timer timer_mpi)
{
    timer_mpi.start();
    HighFive::File file(
        parameters.output_fname, HighFive::File::Overwrite,
        HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    const auto dataspace_embedding = HighFive::DataSpace({df.n_columns()});
    auto dataset_embedding =
        file.createDataSet<uint32_t>("/embedding", dataspace_embedding);
    timer_mpi.stop();

    std::vector<uint32_t> optimal_E(df.n_columns());

    Timer timer;

    auto total_knn_time_simplex = 0.0f;
    auto total_lookup_time_simplex = 0.0f;
    auto total_cpu_to_gpu_time_simplex = 0.0f;
    auto total_gpu_to_cpu_time_simplex = 0.0f;

    if (!rank) {
        std::cout << "Input: " << parameters.input_fname << std::endl;
        std::cout << "Output: " << parameters.output_fname << std::endl;

        timer_mpi.start();
        EmbeddingDimMPIMaster embedding_dim_master(df, MPI_COMM_WORLD);
        timer_mpi.stop();

        Timer timer_embedding_dim;

        timer.start();
        timer_embedding_dim.start();
        embedding_dim_master.run();
        timer_embedding_dim.stop();

        optimal_E = embedding_dim_master.optimal_E;

        dataset_embedding.write(optimal_E);

        std::cout << "Processed optimal E in " << timer_embedding_dim.elapsed()
                  << " [ms]" << std::endl;
    } else {
        if (parameters.kernel_type == "cpu") {
            EmbeddingDimMPIWorker<EmbeddingDimCPU> embedding_dim_worker(
                df, parameters.verbose, MPI_COMM_WORLD);

            embedding_dim_worker.run();
            total_knn_time_simplex = embedding_dim_worker.total_knn_time();
            total_lookup_time_simplex = embedding_dim_worker.total_lookup_time();
            total_cpu_to_gpu_time_simplex = 0;
            total_gpu_to_cpu_time_simplex = 0;
        }
#ifdef ENABLE_GPU_KERNEL
        if (parameters.kernel_type == "gpu") {
            EmbeddingDimMPIWorker<EmbeddingDimGPU> embedding_dim_worker(
                df, parameters.verbose, MPI_COMM_WORLD);

            embedding_dim_worker.run();
            total_knn_time_simplex = embedding_dim_worker.total_knn_time();
            total_lookup_time_simplex = embedding_dim_worker.total_lookup_time();
            total_cpu_to_gpu_time_simplex = embedding_dim_worker.total_cpu_to_gpu_time();
            total_gpu_to_cpu_time_simplex = embedding_dim_worker.total_gpu_to_cpu_time();
        }
#endif
    }

    auto max_knn_time_simplex = 0.0f;
    auto max_lookup_time_simplex = 0.0f;
    auto max_cpu_to_gpu_time_simplex = 0.0f;
    auto max_gpu_to_cpu_time_simplex = 0.0f;

    timer_mpi.start();

    MPI_Reduce(&total_knn_time_simplex, &max_knn_time_simplex, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&total_lookup_time_simplex, &max_lookup_time_simplex, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&total_cpu_to_gpu_time_simplex, &max_cpu_to_gpu_time_simplex, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&total_gpu_to_cpu_time_simplex, &max_gpu_to_cpu_time_simplex, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (!rank) {
        std::cout << "Max kNN (Simplex): " << max_knn_time_simplex << " [ms] | "
                  << "Max Lookup (Simplex): " << max_lookup_time_simplex << " [ms] | "
                  << "Max CPU to GPU (Simplex): " << max_cpu_to_gpu_time_simplex << " [ms] | "
                  << "Max GPU to CPU (Simplex): " << max_gpu_to_cpu_time_simplex << " [ms]"
                  << std::endl;
    }

    MPI_Bcast(optimal_E.data(), optimal_E.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    const auto dataspace_corrcoef =
        HighFive::DataSpace({df.n_columns(), df.n_columns()});
    auto dataset_corrcoef =
        file.createDataSet<float>("/corrcoef", dataspace_corrcoef);

    timer_mpi.stop();

    auto total_io_time = 0.0f;
    auto total_knn_time_cm = 0.0f;
    auto total_lookup_time_cm = 0.0f;
    auto total_cpu_to_gpu_time_cm = 0.0f;
    auto total_gpu_to_cpu_time_cm = 0.0f;

    if (!rank) {
        timer_mpi.start();
        CrossMappingMPIMaster cross_mapping_master(df, parameters.chunk_size,
                                                   MPI_COMM_WORLD);
        timer_mpi.stop();

        Timer timer_cross_mapping;

        timer_cross_mapping.start();
        cross_mapping_master.run();
        timer_cross_mapping.stop();

        timer.stop();

        std::cout << "Cross Mapping " << timer_cross_mapping.elapsed() << " [ms]" << std::endl;
        std::cout << "MPI Time " << timer_mpi.elapsed() << " [ms]" << std::endl;
        std::cout << "Processed dataset in " << timer.elapsed() << " [ms]"
                  << std::endl;
    } else {
        if (parameters.kernel_type == "cpu") {
            CrossMappingMPIWorker<CrossMappingCPU> cross_mapping_worker(
                dataset_corrcoef, df, optimal_E, parameters.verbose,
                MPI_COMM_WORLD);

            cross_mapping_worker.run();

            total_io_time = cross_mapping_worker.total_io_time();
            total_knn_time_cm = cross_mapping_worker.total_knn_time();
            total_lookup_time_cm = cross_mapping_worker.total_lookup_time();
            total_cpu_to_gpu_time_cm = 0;
            total_gpu_to_cpu_time_cm = 0;
        }
#ifdef ENABLE_GPU_KERNEL
        if (parameters.kernel_type == "gpu") {
            CrossMappingMPIWorker<CrossMappingGPU> cross_mapping_worker(
                dataset_corrcoef, df, optimal_E, parameters.verbose,
                MPI_COMM_WORLD);

            cross_mapping_worker.run();

            total_io_time = cross_mapping_worker.total_io_time();
            total_knn_time_cm = cross_mapping_worker.total_knn_time();
            total_lookup_time_cm = cross_mapping_worker.total_lookup_time();
            total_cpu_to_gpu_time_cm = cross_mapping_worker.total_cpu_to_gpu_time();
            total_gpu_to_cpu_time_cm = cross_mapping_worker.total_gpu_to_cpu_time();
        }
#endif
    }

    auto max_io_time = 0.0f;
    auto max_knn_time_cm = 0.0f;
    auto max_lookup_time_cm = 0.0f;
    auto max_cpu_to_gpu_time_cm = 0.0f;
    auto max_gpu_to_cpu_time_cm = 0.0f;

    MPI_Reduce(&total_io_time, &max_io_time, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&total_knn_time_cm, &max_knn_time_cm, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&total_lookup_time_cm, &max_lookup_time_cm, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&total_cpu_to_gpu_time_cm, &max_cpu_to_gpu_time_cm, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&total_gpu_to_cpu_time_cm, &max_gpu_to_cpu_time_cm, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (!rank) {
        std::cout << "Max output IO Time: " << max_io_time << " [ms] | "
                  << "Max kNN (Cross Mapping): " << max_knn_time_cm << " [ms] | "
                  << "Max Lookup (Cross Mapping): " << max_lookup_time_cm << " [ms] | "
                  << "Max CPU to GPU (Cross Mapping): " << max_cpu_to_gpu_time_cm << " [ms] | "
                  << "Max GPU to CPU (Cross Mapping): " << max_gpu_to_cpu_time_cm << " [ms]"
                  << std::endl;
    }
}

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": Cross Mapping Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...] INPUT OUTPUT\n"
        "  -t, --tau arg        Lag (default: 1)\n"
        "  -e, --maxe arg       Maximum embedding dimension (default: 20)\n"
        "  -p, --Tp arg         Steps to predict in future (default: 1)\n"
        "  -x, --kernel arg     Kernel type {cpu|gpu} (default: cpu)\n"
        "  -d, --dataset arg    HDF5 dataset name\n"
        "  -c, --chunksize arg  Number of timeseries per task (default: 1)\n"
        "  -v, --verbose        Enable verbose logging (default: false)\n"
        "  -h, --help           Show help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-t", "--tau", "-p", "--tp", "-e", "--maxe", "-x",
                       "--kernel", "-d", "--dataset", "-c", "--chunksize", "-v",
                       "--verbose"});
    cmdl.parse(argc, argv);

    if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
    }

    if (!cmdl(1)) {
        std::cerr << "No input file" << std::endl;
        usage(cmdl[0]);
        return 1;
    }

    if (!cmdl(2)) {
        std::cerr << "No output file" << std::endl;
        usage(cmdl[0]);
        return 1;
    }

    Parameters parameters;

    parameters.input_fname = cmdl[1];
    parameters.output_fname = cmdl[2];

    cmdl({"t", "tau"}, 1) >> parameters.tau;
    cmdl({"p", "Tp"}, 1) >> parameters.Tp;
    cmdl({"e", "maxe"}, 20) >> parameters.max_E;
    cmdl({"x", "kernel"}, "cpu") >> parameters.kernel_type;
    cmdl({"d", "dataset"}) >> parameters.dataset_name;
    cmdl({"c", "chunksize"}, 1) >> parameters.chunk_size;
    parameters.verbose = cmdl[{"v", "verbose"}];

    Timer timer_mpi;

    timer_mpi.start();
    MPI_Init(&argc, &argv);
    timer_mpi.stop();

    if (argc < 2) {
        std::cerr << "No input" << std::endl;
        return -1;
    }

    timer_mpi.start();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    timer_mpi.stop();

    DataFrame df;

    Timer timer_io;
    timer_io.start();

    if (ends_with(parameters.input_fname, ".csv")) {
        df.load_csv(parameters.input_fname);
    } else if (ends_with(parameters.input_fname, ".hdf5") ||
               ends_with(parameters.input_fname, ".h5")) {
        if (parameters.dataset_name.empty()) {
            std::cerr << "No HDF5 dataset name" << std::endl;
            usage(cmdl[0]);
            return 1;
        }

        df.load_hdf5(parameters.input_fname, parameters.dataset_name);
    } else {
        std::cerr << "Unknown file type" << std::endl;
        usage(cmdl[0]);
        return 1;
    }

    timer_io.stop();

    if (!rank) {
        std::cout << "Read input dataset (" << df.n_rows() << " rows, "
                  << df.n_columns() << " columns) in " << timer_io.elapsed()
                  << " [ms]" << std::endl;
    }

    run(rank, df, parameters, timer_mpi);

    MPI_Finalize();
}
