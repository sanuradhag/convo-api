model = {
    "interactionModel": {
        "languageModel": {
            "invocationName": "cake walk",
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
                },
                {
                    "name": "CaptureBirthdayIntent",
                    "slots": [
                        {
                            "name": "month",
                            "type": "AMAZON.Month"
                        },
                        {
                            "name": "day",
                            "type": "AMAZON.Ordinal"
                        },
                        {
                            "name": "year",
                            "type": "AMAZON.FOUR_DIGIT_NUMBER"
                        }
                    ],
                    "samples": [
                        "I was born on {month}  {year}",
                        "I was born on {month} {day} ",
                        "{month} {year}",
                        "{month} {day}",
                        "{month} {day} {year}",
                        "I was born on {month} {day} {year}"
                    ]
                }
            ],
            "types": []
        },
        "dialog": {
            "intents": [
                {
                    "name": "CaptureBirthdayIntent",
                    "confirmationRequired": 'false',
                    "prompts": {},
                    "slots": [
                        {
                            "name": "month",
                            "type": "AMAZON.Month",
                            "confirmationRequired": 'false',
                            "elicitationRequired": 'true',
                            "prompts": {
                                "elicitation": "Elicit.Slot.750741312188.566599050269"
                            }
                        },
                        {
                            "name": "day",
                            "type": "AMAZON.Ordinal",
                            "confirmationRequired": 'false',
                            "elicitationRequired": 'true',
                            "prompts": {
                                "elicitation": "Elicit.Slot.750741312188.785139915502"
                            }
                        },
                        {
                            "name": "year",
                            "type": "AMAZON.FOUR_DIGIT_NUMBER",
                            "confirmationRequired": 'false',
                            "elicitationRequired": 'true',
                            "prompts": {
                                "elicitation": "Elicit.Slot.750741312188.1167430023377"
                            }
                        }
                    ]
                }
            ],
            "delegationStrategy": "ALWAYS"
        },
        "prompts": [
            {
                "id": "Elicit.Slot.750741312188.566599050269",
                "variations": [
                    {
                        "type": "PlainText",
                        "value": "What month were you born in?"
                    }
                ]
            },
            {
                "id": "Elicit.Slot.750741312188.785139915502",
                "variations": [
                    {
                        "type": "PlainText",
                        "value": "What day were you born?"
                    }
                ]
            },
            {
                "id": "Elicit.Slot.750741312188.1167430023377",
                "variations": [
                    {
                        "type": "PlainText",
                        "value": "What year were you born?"
                    }
                ]
            }
        ]
    }
}