import json
from collections import namedtuple


class InteractionModelGenerator:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
        self.jsn_str = ""
        self.spec = ""
        self.intents = []
        self.parsed_json = ""
        self.default_slots = ['AMAZON.DATE', 'AMAZON.DURATION', 'AMAZON.FOUR_DIGIT_NUMBER', 'AMAZON.NUMBER',
                              'AMAZON.Ordinal', 'AMAZON.PhoneNumber', 'AMAZON.TIME', 'AMAZON.SearchQuery']
        self.spec_intents = []
        self.intraction_model = {
            "interactionModel": {
                "languageModel": {
                    "invocationName": "",
                    "intents": [
                        {
                            "name": "AMAZON.CancelIntent",
                            "samples": []
                        },
                        {
                            "name": "AMAZON.HelpIntent",
                            "samples": []
                        },
                        {
                            "name": "AMAZON.StopIntent",
                            "samples": []
                        },
                        {
                            "name": "AMAZON.NavigateHomeIntent",
                            "samples": []
                        }
                    ]
                },
                "dialog": {
                    "intents": []
                }
            }
        }

    def generate(self):
        self.open_spec_file()
        paths = self.parsed_json['paths']
        for path in paths:
            current_path = paths[path]
            for method in current_path:
                current_method = current_path[method]
                if (method == 'post') | (method == 'put'):
                    summary = current_method['summary']
                    operation_id = current_method['operationId']
                    if 'requestBody' in current_method:
                        request_body = current_method['requestBody']
                        content = request_body['content']
                        if 'application/json' in content:
                            schema_name = \
                                current_method['requestBody']['content']['application/json']['schema']['$ref'].split(
                                    '/')[
                                    -1]
                            self.generate_intents(schema_name, operation_id)
        print('spec_intents', self.spec_intents)
        print('\n')

        # set generated intents
        self.intraction_model["interactionModel"]["languageModel"]["intents"] = \
            self.intraction_model["interactionModel"]["languageModel"]["intents"] + self.spec_intents

        # set invocation name
        self.intraction_model["interactionModel"]["languageModel"]["invocationName"] = self.parsed_json['info']['title']

        print('interaction_model', self.intraction_model)
        return self.intraction_model

    def open_spec_file(self):
        path_to_file = self.path + '/' + self.filename
        try:
            with open(path_to_file) as f:
                data = json.load(f)
                self.parsed_json = data
                self.jsn_str = json.dumps(data, indent=4)
        except json.decoder.JSONDecodeError as e:
            print("Error occurred while opening the spec file")
            print(e)

    def generate_intents(self, schema_name, intent_name):
        slots = []
        components = self.parsed_json['components']
        schemas = components["schemas"]
        current_schema = schemas[schema_name]
        props = current_schema['properties']
        for prop in props:
            current_prop = props[prop]
            if 'type' in current_prop:
                type = current_prop['type']
                slot = self.map_to_slot_type(type)
                if slot is not None:
                    slots.append(slot)
        # slots = list(dict.fromkeys(slots))
        slots = self.remove_duplicates(slots)
        self.spec_intents.append({"name": intent_name, "slots": slots, "samples": []})

    def map_to_slot_type(self, t):
        if (t == "integer") | (t == "number"):
            return {"name": "number", "type": "AMAZON.NUMBER"}
        elif t == "string":
            return {"name": "string", "type": "AMAZON.Ordinal"}
        else:
            return None

    def remove_duplicates(self, list):
        return [dict(t) for t in {tuple(d.items()) for d in list}]