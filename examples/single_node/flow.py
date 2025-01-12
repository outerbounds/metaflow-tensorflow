from metaflow import FlowSpec, step, kubernetes, environment, pypi
from gpu_profile import gpu_profile


class SingleNodeTensorFlow(FlowSpec):
    local_model_dir = "model"
    local_tar_name = "mnist.tar.gz"

    @step
    def start(self):
        self.next(self.train)

    @gpu_profile(interval=1)
    @environment(vars={"TF_CPP_MIN_LOG_LEVEL": "2"})
    @kubernetes(gpu=2, image="registry.hub.docker.com/tensorflow/tensorflow:2.15.0-gpu")
    @pypi(
        packages={
            "tensorflow-datasets": "4.9.7",
            "matplotlib": "3.10.0",
        }
    )
    @step
    def train(self):
        from train_mnist import main

        main(
            local_model_dir=self.local_model_dir,
            local_tar_name=self.local_tar_name,
            run=self,
        )

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SingleNodeTensorFlow()
