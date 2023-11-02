from metaflow import FlowSpec, step, batch, conda, environment

N_GPU = 2


class SingleNodeTensorFlow(FlowSpec):
    local_model_dir = "model"
    local_tar_name = "mnist.tar.gz"

    @step
    def start(self):
        self.next(self.foo)

    @environment(vars={"TF_CPP_MIN_LOG_LEVEL": "2"})
    @batch(gpu=N_GPU, image="tensorflow/tensorflow:latest-gpu")
    @step
    def foo(self):
        from mnist_mirrored_strategy import main

        main(
            run=self,
            local_model_dir=self.local_model_dir,
            local_tar_name=self.local_tar_name,
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SingleNodeTensorFlow()
