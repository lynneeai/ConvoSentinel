import json
import pathlib
from openai import OpenAI, BadRequestError
from openai import AzureOpenAI


client = OpenAI(api_key=open("../openai_api_key.txt", "r").read().strip())

pricing = {
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06}
}


def call_gpt(prompt, system_message=None, model="gpt-3.5-turbo", max_tokens=512, response_format=None):
    """
    Calls the OpenAI model to generate a completion based on the given prompt.

    Args:
        prompt (str): The user's prompt for the model.
        system_message (str, optional): The system message to be shown before the user's prompt. Defaults to None.
        model (str, optional): The model to use for completion. Defaults to "gpt-3.5-turbo".
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
        response_format (dict, optional): The format of the response. Defaults to None.

    Returns:
        str: The generated completion.
    """
    messages = [{"role": "user", "content": prompt}]
    if system_message:
        messages = [{"role": "system", "content": system_message}] + messages
    
    params = {
        "model": model,
        "messages": messages,
        "temperature": 1,
        "max_tokens": max_tokens,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    if response_format:
        params["response_format"] = response_format
        
    response = client.chat.completions.create(**params)
    usage = response.usage
    
    return response.choices[0].message.content, get_cost(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, model=model)


def call_gpt_json_response_wrapper(prompt, must_have_fields, system_message=None, model="gpt-3.5-turbo", max_tokens=256, max_try=3):
    """
    Calls the OpenAI model to generate a completion based on the given prompt and returns the response in JSON format.

    Args:
        prompt (str): The user's prompt for the model.
        must_have_fields (list): A list of fields that must be present in the generated JSON response.
        model (str, optional): The model to use for completion. Defaults to "gpt-3.5-turbo".
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.
        max_try (int, optional): The maximum number of attempts to generate a valid JSON response. Defaults to 3.

    Returns:
        dict: A dictionary containing the generated JSON response.
            The dictionary has the following keys:
            - "succeeded" (bool): Indicates whether the generation was successful or not.
            - Additional fields specified in the `must_have_fields` list.
    """
    total_cost = {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for i in range(max_try):
        try:
            output, cost = call_gpt(prompt, system_message=system_message, model=model, response_format={"type": "json_object"}, max_tokens=max_tokens)
            for k, v in cost.items():
                total_cost[k] += v
            output = json.loads(output)
            assert all(field in output for field in must_have_fields)
            return output, total_cost
        except BadRequestError as e:
            raise e
        except:
            continue
    return None, total_cost


def get_cost_estimate(input_word_count, model, output_word_count=None):
    """
    Calculates the estimated cost of generating a completion based on the given prompt.

    Args:
        input_word_count (int): The number of words in the prompt.
        model (str): The model to use for completion.
        output_word_count (int, optional): The number of words in the generated completion. Defaults to None.

    Returns:
        dict: A dictionary containing the cost estimate and token counts.
            The dictionary has the following keys:
            - "cost" (float): The estimated cost in dollars.
            - "prompt_tokens" (float): The number of tokens in the prompt.
            - "completion_tokens" (float): The number of tokens in the generated completion, if provided.
    """
    prompt_tokens = input_word_count * 4.0 / 3
    total_cost = prompt_tokens * pricing[model]["input"] / 1000
    
    if output_word_count:
        completion_tokens = output_word_count * 4.0 / 3
        total_cost += completion_tokens * pricing[model]["output"] / 1000
    
    return {"cost": total_cost, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens if output_word_count else 0}


def get_cost(prompt_tokens, completion_tokens, model):
    """
    Calculates the estimated cost of generating a completion based on the given token counts.

    Args:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the generated completion.
        model (str): The model to use for completion.

    Returns:
        float: The estimated cost in dollars.
    """
    total_cost = prompt_tokens * pricing[model]["input"] / 1000
    total_cost += completion_tokens * pricing[model]["output"] / 1000
    return {"cost": total_cost, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}