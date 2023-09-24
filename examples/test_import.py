from metaflow import FlowSpec, step, kubernetes, conda, tensorflow_parallel

class TestImport(FlowSpec):

    @step
    def start(self):
        self.next(self.foo, num_parallel=2)

    @conda(libraries={"tensorflow": "2.12.1"})
    @kubernetes # comment this out and ensure it runs locally too.
    @tensorflow_parallel
    @step
    def foo(self):
        import tensorflow as tf
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    TestImport()