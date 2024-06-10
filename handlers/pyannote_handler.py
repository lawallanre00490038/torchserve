# # custom handler file

# # model_handler.py

# """
# ModelHandler defines a custom model handler.
# """

# from ts.torch_handler.base_handler import BaseHandler
# import torch, os

# class ModelHandler(BaseHandler):
#     """
#     A custom model handler implementation.
#     """

#     def __init__(self):
#         self._context = None
#         self.initialized = False
#         self.explain = False
#         self.target = 0

#     def initialize(self, context):
#         """
#         Invoke by torchserve for loading a model
#         :param context: context contains model server system properties
#         :return:
#         """

#         #  load the model
#         self.manifest = context.manifest

#         properties = context.system_properties
#         model_dir = properties.get("model_dir")
#         self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

#         # Read model serialize/pt file
#         serialized_file = self.manifest['model']['serializedFile']
#         model_pt_path = os.path.join(model_dir, serialized_file)
#         if not os.path.isfile(model_pt_path):
#             raise RuntimeError("Missing the model.pt file")

#         self.model = torch.jit.load(model_pt_path)

#         self.initialized = True

#     def preprocess(self, data):
#         """
#         Transform raw input into model input data.
#         :param batch: list of raw requests, should match batch size
#         :return: list of preprocessed model input data
#         """
#         # Take the input data and make it inference ready
#         preprocessed_data = data[0].get("data")
#         if preprocessed_data is None:
#             preprocessed_data = data[0].get("body")

#         return preprocessed_data


#     def inference(self, model_input):
#         """
#         Internal inference methods
#         :param model_input: transformed model input data
#         :return: list of inference output in NDArray
#         """
#         # Do some inference call to engine here and return output
#         model_output = self.model.forward(model_input)
#         return model_output

#     def postprocess(self, inference_output):
#         """
#         Return inference result.
#         :param inference_output: list of inference output
#         :return: list of predict results
#         """
#         # Take output from network and post-process to desired format
#         postprocess_output = inference_output
#         return postprocess_output

#     def handle(self, data, context):
#         """
#         Invoke by TorchServe for prediction request.
#         Do pre-processing of data, prediction using model and postprocessing of prediciton output
#         :param data: Input data for prediction
#         :param context: Initial context contains model server system properties.
#         :return: prediction output
#         """
#         model_input = self.preprocess(data)
#         model_output = self.inference(model_input)
#         return self.postprocess(model_output)


from pyannote.audio import Pipeline, Audio
import torch


class EndpointHandler:
    def __init__(self, path=""):
        # initialize pretrained pipeline

        model_pt_path = "model_dir/pyannote_diarization.pt"
        self.model = torch.jit.load(model_pt_path)
        #self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        # send pipeline to GPU if available
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))

        # initialize audio reader
        self._io = Audio()

    def __call__(self, data):
        inputs = data.pop("inputs", data)
        waveform, sample_rate = self._io(inputs)

        parameters = data.pop("parameters", dict())
        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}, **parameters
        )

        processed_diarization = [
            {
                "speaker": speaker,
                "start": f"{turn.start:.3f}",
                "end": f"{turn.end:.3f}",
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        return {"diarization": processed_diarization}