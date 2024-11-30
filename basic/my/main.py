import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Initialize an empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Save tasks to a CSV file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Train the task priority classifier
def train_model():
    if not tasks.empty:
        vectorizer = CountVectorizer()
        clf = MultinomialNB()
        model = make_pipeline(vectorizer, clf)
        model.fit(tasks['description'], tasks['priority'])
        return model
    return None

model = train_model()

# Add a task to the list
def add_task(description, priority):
    global tasks
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()
    print("Task added successfully.")
    return train_model()  # Retrain model

# Remove a task by description
def remove_task(description):
    global tasks
    if description in tasks['description'].values:
        tasks = tasks[tasks['description'] != description]
        save_tasks()
        print("Task removed successfully.")
    else:
        print("Task not found.")

# List all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print("\nTasks:\n", tasks)

# Recommend a task based on machine learning
def recommend_task():
    if not tasks.empty:
        if model:
            high_priority_tasks = tasks[tasks['priority'] == 'High']
            if not high_priority_tasks.empty:
                random_task = random.choice(high_priority_tasks['description'].values)
                print(f"Recommended task: {random_task} - Priority: High")
            else:
                print("No high-priority tasks available for recommendation.")
        else:
            print("Not enough data to make recommendations.")
    else:
        print("No tasks available for recommendations.")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        model = add_task(description, priority)

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
