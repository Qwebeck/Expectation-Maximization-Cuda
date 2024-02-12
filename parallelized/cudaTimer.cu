struct cuTimer
{
    cudaEvent_t _start;
    cudaEvent_t _end;
    float elapsedTime;

    void start()
    {
        cudaEventCreate(&_start);
        cudaEventCreate(&_end);
        cudaEventRecord(_start);
    }

    void stop()
    {
        cudaEventRecord(_end);
        cudaEventSynchronize(_end);
        cudaEventElapsedTime(&elapsedTime, _start, _end);
    }

    ~cuTimer()
    {
        cudaEventDestroy(_start);
        cudaEventDestroy(_end);
    }
};