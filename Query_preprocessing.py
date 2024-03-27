import pandas as pd

from openai import OpenAI as OpenAI_original
import matplotlib.pyplot as plt
import re
import os
import DataCleaning as dc
import json
import MachineLearning as ml
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI as OpenAI_langchain
import Gptapi as gptapi
import faiss


def parse_instructions1(instructions):
    parsed = {}
    # Split by commas that are not inside parentheses
    parts = re.split(r',(?![^()]*\))', instructions)
    for part in parts:
        # Split by equal signs that are not inside parentheses
        key_value = re.split(r':(?![^()]*\))', part)
        if len(key_value) == 2:
            key, value = key_value
            parsed[key.strip().lower()] = value.strip()
    return parsed


def parse_instructions2(instruction_str):
    # Parse the JSON string into a Python dictionary
    instructions = json.loads(instruction_str)

    # Extract the "standards" and "variations" from the instructions
    standards = instructions.get('standards', {})
    variations = instructions.get('StringMatching', {})

    return standards, variations


def parse_instructions3(instructions):
    # Split the instructions by commas
    parts = re.split(',(?![^{}]*\})', instructions)

    # Initialize an empty dictionary to store the parsed instructions
    parsed = {}

    for part in parts:
        # Split each part by the colon
        key_value = part.split(':')

        if len(key_value) == 2:
            key, value = key_value

            # Remove any leading or trailing whitespace
            key = key.strip()
            value = value.strip()

            # If the value is enclosed in curly braces, it's a list of features
            if value.startswith('{') and value.endswith('}'):
                # Remove the curly braces and split by comma
                value = value[1:-1].split(',')

                # Remove any leading or trailing whitespace from each feature
                value = [v.strip() for v in value]

            # Add the key-value pair to the dictionary
            parsed[key] = value

    return parsed


def apply_conditions_and_filter(df, x_column, y_column, condition):
    if condition == 'null':
        # Filter rows where both X and Y are non-null
        filtered_df = df.dropna(subset=[x_column, y_column], how='all')
    else:
        fields = re.split(r'[<>=]', condition)
        if len(fields) == 2:
            field, value = fields
            # Apply the condition
            if '=' in condition:
                if df[field].dtype == 'object':
                    filtered_df = df[df[field] == value]
                else:
                    try:
                        value = float(value)
                        filtered_df = df[df[field] == value]
                    except ValueError:
                        print(
                            f"Invalid condition: {condition}. Value {value} cannot be converted to float.")
                        filtered_df = df

                print(filtered_df.head(10))
            else:
                try:
                    value = float(value)
                    if '<' in condition:
                        filtered_df = df[df[field] < value]
                    elif '>' in condition:
                        filtered_df = df[df[field] > value]
                    elif '<=' in condition:
                        filtered_df = df[df[field] <= value]
                    elif '>=' in condition:
                        filtered_df = df[df[field] >= value]
                except ValueError:
                    print(
                        f"Invalid condition: {condition}. Value {value} cannot be converted to float.")
                    filtered_df = df
        else:
            filtered_df = df
        # Dynamic condition application; this example handles simple 'null' condition only

    # Further filtering or processing based on more complex conditions can be added here

    return filtered_df


def plot_data(filtered_df, x_column, y_column):
    if not filtered_df.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_df[x_column], filtered_df[y_column], alpha=0.5)
        plt.title(f'{y_column} vs. {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True)
        plt.show()
    else:
        print("No data to plot after applying the condition.")


if __name__ == "__main__":
    api_key = os.getenv('API_KEY')
    # Load your OpenAI API Key
    # openai.api_key =
    # Initialize the OpenAI client
    client = OpenAI_original(api_key=api_key)

    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv('application_record.csv')
    print("DataFrame loaded.")

    # Collect column names for debugging
    column_names = df.columns.tolist()
    print(f"Column names: {column_names}")

    # Identify categorical columns and their unique values
    categorical_info = []
    for column in df.columns:
        unique_values = df[column].unique()
        if df[column].dtype == 'object' or set(unique_values) == {0, 1} or df[column].nunique() < 10:
            categorical_info.append(
                f"{column}: {', '.join(map(str, unique_values))}")

    # Convert categorical_info to a single string for the prompt
    categorical_info_str = " | ".join(categorical_info)
    df_head_str = df.head().to_string()
    summary = df.describe(include='all')
    summary = summary.to_string()
    print(f"Summary: {summary}")
    context = f"You are given a dataset with columns: {', '.join(column_names)}\n, and the following categorical values: {categorical_info_str}\n\n. The head of the dataset is as follows:\n{df_head_str}\n\n, the data types of the columns are: {','.join(map(str, df.dtypes.tolist()))}\n\n, and the summary of the dataset is as follows:\n{summary}. \n"
    # Specify the behavior of the AI assistant in the system message
    system_message = {
        "role": "system",
        "content": "You are a helpful data analyst who understands data analysis and can provide instructions for processing data to answer specific questions. You can help with tasks such as Data Cleaning, Trend Analysis, and Machine Learning tasks, and Talk about the data. You can also handle filtering based on conditions. You can provide instructions in a structured format."
    }

    while True:

        # Specify the query with the column names included
        query = input("Enter your query: ")
        if query.lower() == 'quit':
            break

        user_message = {
            "role": "user",
            "content": (f"Given a dataset with columns {', '.join(column_names)}, "
                        f"and the following categorical values: {', '.join(categorical_info)}, "
                        f"how should the data be processed to answer the question: '{query}'? "
                        "Provide instructions including Y(always numeric)(row, null) target, X(categorical or numeric)(column, null) target, "
                        "Condition for filtering data, and the type of analysis (GPT Analysis, "
                        "Trend Analysis, Machine Learning tasks, and Data Cleaning). Sample responses: {{Y(always numeric)(row, null):ColumnIncome, X(categorical or numeric)(column, null):ColumnFamilySize, ColumnOwnedHouse, Condition(field<n,field>n,field=n,field<=n, field>=n,null):DaysBirth<-3000, TypeOfAnalysis:Trend Analysis}}{{Y(always numeric)(row, null):Null, X(categorical or numeric)(column, null):Null, Condition(field<n,field>n,field=n,field<=n, field>=n,null):Null, TypeOfAnalysis:GPT Analysis}}. Format your response exactly as follows in one single line where you have to replace \"<insert here>\" with the appropriate value without any full stop, explanation and curly braces: "
                        "Y(always numeric)(row, null):<insert here>, X(categorical or numeric)(column, null):<insert here>, Condition(field<n,field>n,field=n,field<=n, field>=n,null):<insert here>, TypeOfAnalysis:<insert here>")
        }

        # Query GPT-3.5 Turbo
        chat_completion = client.chat.completions.create(
            messages=[system_message, user_message],
            model="gpt-3.5-turbo",
        )

        # Extract and print the structured instructions from the response
        instructions = chat_completion.choices[0].message.content
        print(f"GPT-3.5 Turbo Instructions: {instructions}\n")

        # Extracting parameters from the instructions
        params = parse_instructions1(instructions)
        y_column = params.get('y(always numeric)(row, null)',
                              '').replace('null', '').strip()
        x_column = params.get('x(categorical or numeric)(column, null)', '').replace(
            'null', '').strip()
        # Defaulting to 'null' if not specified
        condition = 'null'
        for key in params:
            if key.startswith('condition'):
                condition = params.get(key, 'null')
                break

        typeofanalysis = params.get('typeofanalysis', '').strip()
        print(f"type of analysis: {typeofanalysis}")

        if typeofanalysis == 'Trend Analysis':
            # Apply trend analysis
            # Applying conditions and filtering the DataFrame
            filtered_df = apply_conditions_and_filter(
                df, x_column, y_column, condition)

            # Plotting the data
            plot_data(filtered_df, x_column, y_column)

        elif typeofanalysis == 'Data Cleaning':
            # Apply data cleaning based on how GPT can recognize different representations of the same category(e.g., "USA" and "United States") into a standard format

            # print(f"Categorical info: {categorical_info_str}")

            # Now, construct your system and user messages
            system_message = {
                "role": "system",
                "content": "You are a Data analyst. Use the knowledge provided in the prompt to suggest data cleaning standards and variations for columns only when needed."
            }
            df.drop_duplicates(inplace=True)

            print(f"Head of the DataFrame:\n\n{df_head_str}")

            user_message = {
                "role": "user",
                "content": f"You are given a dataset with columns: {', '.join(column_names)}, and the following categorical values: {categorical_info_str}. The head of the dataset is as follows:\n{df_head_str}\nBased on the information provided give standards of only the columns that have actual phone numbers(not flags), dates(not number of days), email address(not flags) if there are any such columns in the data and identify variations that exist in the given columns referring to the same identity. And if there are no variations based on the categorical values of the columns then leave those columns empty in the output. To give standardization and identify variations of only the columns in the given head of the dataset, suggest standards and variations in a properly formatted structured output and don't include the columns that are already standardized or don't have variations, ignore columns that only have numbers; Following is an Example output in JSON:{{\"standards\": {{\"ColumnDateOfBirths\": \"date format\", \"ColumnEmail_Addresses\": \"email format\", \"ColumnPhone_Numbers\": \"phone number format\"}}, \"StringMatching\": {{\"CloumnCompany\": {{\"Google\": [\"Google\", \"Google Inc\", \"Googlle\"],\"Apple\": [\"Apple\", \"Apple Inc.\", \"Aple\"]}}, \"Column4\":{{\"Target2\":[\"variation3\", \"Variation4\"]}}}}}}"
            }

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[system_message, user_message]
            )

            instructions = response.choices[0].message.content
            print(f"GPT-3.5 Turbo Instructions: {instructions}")
            standards, variations = parse_instructions2(instructions)\
                # Print the standards and variations
            print(f"Standards: {standards}")
            if not standards:
                df = dc.Standardize(df, standards)

            if not variations:
                df = dc.StringMatching(df, variations)

            print(f"Variations: {variations}")

        if typeofanalysis == 'Machine Learning tasks':
            # Apply machine learning tasks
            # Apply machine learning tasks based on the data
            # This is just a placeholder for the actual machine learning tasks

            system_message = {
                "role": "system",
                "content": "You are a Data analyst. Use the knowledge provided in the prompt to suggest machine learning tasks based on the data."
            }
            column_types = df.dtypes.tolist()
            user_message = {
                "role": "user",
                "content": f"You are given a dataset with columns: {', '.join(map(str, column_names))}, column data types respectively: {','.join(map(str, column_types))} and categorical values: {categorical_info_str}. Answer the question: '{query}'? Based on the information provided, apply machine learning tasks to the data. Provide instructions including the type of machine learning task (Predictive Classification, Cluster Model, Regression Model), the target variable(Column name), and the features to be used. Format your response exactly as follows in one single line where you have to replace <insert here> with the appropriate value: \"TypeOfMachineLearningTask:<insert here>, TargetVariable:{{<insert here>}}, Features:{{<insert here>, <insert here>}}\""

            }

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[system_message, user_message]
            )

            instructions = response.choices[0].message.content
            # Remove double quotes from the beginning and the end
            instructions = instructions.strip('"')
            print(f"GPT-3.5 Turbo Instructions: {instructions}\n\n")
            parsed = parse_instructions3(instructions)
            ml.apply_model(df, parsed)

            print("Applying machine learning tasks based on the data...")
            # Placeholder for the actual machine learning tasks

        elif typeofanalysis == 'GPT Analysis':
            # Apply all other queries
            print("Applying all other queries based on the data...\n")
            # Placeholder for the actual all other queries
            # system_message = {
            #     "role": "system",
            #     "content": "You are an AI assitant Data analyst. Use the knowledge provided in the prompt and context to answer queries in a conversational manner."
            # }
            system_message = {
                "role": "system",
                "content": "You are a helpful data analyst who understands data analysis and can talk about the data given context and an answer by another LLM."
            }
            # Initialize FAISS index
            dimension = 384  # Dimension of embeddings
            # Use IndexFlatIP for cosine similarity
            index = faiss.IndexFlatL2(dimension)
            chunks = gptapi.split_text(context)
            index = gptapi.index_chunks(chunks, index)
            relevant_chunks = gptapi.query_faiss(index, chunks, query, top_k=5)
            contextnew = ','.join(relevant_chunks)
            user_message = {
                "role": "user",
                "content": f" Question: '{query}'? \n"
            }
            agent = create_csv_agent(
                ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",
                           openai_api_key=api_key),
                "application_record.csv",
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS
            )
            Answer = agent.invoke(f"{query}")
            print(Answer, "\n \n")

            # Convert Answer to string if it's not already a string
            if not isinstance(Answer, str):
                Answer = str(Answer)

            user_message2 = {
                "role": "user",
                "content": f" Context: '{contextnew}'\n Answer: '{Answer}'? \n chat with the user about the answer:"
            }
            context += Answer
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[system_message, user_message2]
            )
            context += response.choices[0].message.content
            print(response.choices[0].message.content)
