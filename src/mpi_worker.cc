#include <iostream>
#include <fstream>

#include "mpi_common.h"
#include "mpi_worker.h"
#include "timer.h"

// Based on https://github.com/nepda/pi-pp/blob/master/serie_4
void MPIWorker::run(int rank, char timer_process_type)
{
    MPI_Status stat;

    while (true) {
        Timer timer_mpi;
        
        timer_mpi.start();
        // Here we send a message to the master asking for a task
        MPI_Send(nullptr, 0, MPI_BYTE, 0, TAG_ASK_FOR_TASK, comm);
        timer_mpi.stop();

        timer_mpi.start();
        // Wait for a reply from master
        MPI_Probe(0, MPI_ANY_TAG, comm, &stat);
        timer_mpi.stop();

        // We got a task
        if (stat.MPI_TAG == TAG_TASK_DATA) {
            auto count = 0;

            timer_mpi.start();
            MPI_Get_count(&stat, MPI_BYTE, &count);
            timer_mpi.stop();

            std::vector<uint8_t> recv_buf(count);

            timer_mpi.start();
            // Retrieve task data from master into msg_buffer
            MPI_Recv(recv_buf.data(), count, MPI_BYTE, 0, TAG_TASK_DATA, comm,
                     &stat);
            timer_mpi.stop();

            // Work on task
            nlohmann::json result;
            do_task(result, nlohmann::json::from_cbor(recv_buf));

            // Send result to master
            const auto send_buf = nlohmann::json::to_cbor(result);
            timer_mpi.start();
            MPI_Send(send_buf.data(), send_buf.size(), MPI_BYTE, 0, TAG_RESULT,
                     comm);
            timer_mpi.stop();

            std::ofstream result_timer;
            if (timer_process_type == 's')
            {
                result_timer.open("timer_simplex_result_" + std::to_string(rank) + ".csv", std::ios_base::app);
            }
            else if (timer_process_type == 'c')
            {
                result_timer.open("timer_crossmap_result_" + std::to_string(rank) + ".csv", std::ios_base::app);
            }
            result_timer << ", " << timer_mpi.elapsed() << std::endl;
            result_timer.close();

            // We got a stop message
        } else if (stat.MPI_TAG == TAG_STOP) {
            MPI_Recv(nullptr, 0, MPI_BYTE, 0, TAG_STOP, comm, &stat);
            break;
        }
    }
}
