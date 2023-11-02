from metaflow import FlowSpec, step, batch, conda, tensorflow, environment

N_NODES = 2
N_GPU = 2


class MultiNodeTensorFlow(FlowSpec):
    tarfile = "mnist.tar.gz"
    local_model_dir = "model"

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @environment(vars={"TF_CPP_MIN_LOG_LEVEL": "2"})
    @batch(gpu=N_GPU, image="tensorflow/tensorflow:latest-gpu")
    @tensorflow
    @step
    def train(self):
        from mnist import main

        main(
            num_workers=N_NODES,
            run=self,
            local_model_dir=self.local_model_dir,
            local_tar_name=self.tarfile,
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MultiNodeTensorFlow()
