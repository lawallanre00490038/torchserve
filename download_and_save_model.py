import torch
from pyannote.audio import Pipeline
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, AutoModelForCausalLM, AutoTokenizer
import argparse

def download_and_save_pyannote_model(model_name: str, save_path: str):
    try:
        pipeline = Pipeline.from_pretrained(model_name)
        torch.save(pipeline, save_path)

        print(f"Model '{model_name}' has been successfully downloaded and saved to '{save_path}'")
    except Exception as e:
        print(f"An error occured while downloading and saving the model '{e}'")


def download_and_save_whisper_model(model_name: str, save_path: str):
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        tokenizer = WhisperTokenizer.from_pretrained(model_name)

        torch.save({
            'model': model.state_dict(),
            'tokenizer': tokenizer
        }, save_path)

        print(f"Model '{model_name}' has been successfully downloaded and saved to '{save_path}'")

    except Exception as e:
        print(f"An error occured while downloading or saving the model: {e}")

def download_and_save_mistral_model(model_name: str, save_path: str, token: str = None):
    try:
        if token:
            model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
            tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
 
        torch.save({
            'model': model,
            'tokenizer': tokenizer
        }, save_path)

        print(f"Model '{model_name}' has been successfully downloaded and saved to '{save_path}'")

    except Exception as e:
        print(f"An error occured while downloading or saving the model: '{e}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and save model from Hugging Face')
    parser.add_argument('--model_name', type=str, required=True, help='The name of the model to download from Hugging Face')
    parser.add_argument('--save_path', type=str, required=True, help='The path to save the serialized model')
    parser.add_argument('--access_token', type=str, required=False, help='The Hugging Face access token for gated models')
    parser.add_argument('--model_type', type=str, choices=['pyannote', 'whisper', 'mistral'], required=True, help='The of model to download (pyannote or whisper)')

    args = parser.parse_args()

    if args.model_type == 'pyannote':
        download_and_save_pyannote_model(args.model_name, args.save_path)

    elif args.model_type =='whisper':
        download_and_save_whisper_model(args.model_name, args.save_path)

    elif args.model_type == 'mistral':
        download_and_save_mistral_model(args.model_name, args.save_path)