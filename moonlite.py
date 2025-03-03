# impport necassary libarries

# define a fuction to handle CLI input

    # set up arguemnt parser

    # add argumnet for query input

    # parse the arugment s

# define a fuction to load vecotr database

    # check if databsae exists

    # if not, creat it

    # return databse connction

# define a fuction to process querries

    # take user inpt

    #convrt input into vectoor embedding

    # serch for closest matcing results in db

    # retrun results

# define main fuction

    # check if user proivded a querry

    # if so, process it

    # else , show hlp msg

# excute main fuction if script is run direcly

# i am building a cli rag system which means this program will take a query from the user, search for relevant information and return useful results
    # since everything in the program depends on what the user asks for, handling user input seems to be a good place to start
    # i will structure the input first creating an entry point for the system, if we dont handle input correctly, nothing else can function properly. 
# user inputs query, 

import argparse

def process_input():
    parser = argparse.ArgumentParser(description="Moonlite Systems CLI")
    parser.add_argument("--query", "-q", type=str, help="Enter query")
    args = parser.parse_args()

    cleaned_input = args.query.strip().lower() if args.query else None

    if cleaned_input == "exit":
        print("Exiting...")
        return None

    return cleaned_input

if __name__ == "__main__":
    user_query = process_input()
    if user_query:
        print(f"Received query: {user_query}")