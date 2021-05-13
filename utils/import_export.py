import codecs
import json


def dump_model(model_dict, file_encoder, filepath='model'):
    with open(f"{filepath.replace('.json', '')}.json", 'w') as fp:
        json.dump(model_dict, fp, default=file_encoder)


def load_model(filepath='model'):
    helper_filepath = filepath if filepath.endswith('.json') else f"{filepath}.json"
    file_text = codecs.open(helper_filepath, 'r', encoding='utf-8').read()
    model_json = json.loads(file_text)

    return model_json