import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set font for displaying Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_power_control_record(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create a large figure
    plt.figure(figsize=(20, 15))
    
    # 1. Node states distribution over time
    plt.subplot(3, 1, 1)
    plt.plot(df['datetime'], df['nb_computing'], label='Computing', color='green')
    plt.plot(df['datetime'], df['nb_idle'], label='Idle', color='blue')
    plt.plot(df['datetime'], df['nb_sleeping'], label='Sleeping', color='orange')
    plt.plot(df['datetime'], df['nb_powered_off'], label='Powered Off', color='red')
    plt.title('Node States Distribution Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Nodes')
    plt.legend()
    plt.grid(True)
    
    # 2. Jobs status over time
    plt.subplot(3, 1, 2)
    plt.plot(df['datetime'], df['running_jobs'], label='Running Jobs', color='blue')
    plt.plot(df['datetime'], df['waiting_jobs'], label='Waiting Jobs', color='red')
    plt.title('Jobs Status Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Jobs')
    plt.legend()
    plt.grid(True)
    
    # 3. Utilization rate over time
    plt.subplot(3, 1, 3)
    plt.plot(df['datetime'], df['utilization_rate'], label='Utilization Rate', color='green')
    plt.title('Resource Utilization Over Time')
    plt.xlabel('Time')
    plt.ylabel('Utilization Rate (%)')
    plt.legend()
    plt.grid(True)
    
    # Adjust subplot spacing
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('power_control_analysis.png')
    
    # Calculate statistics
    stats = {
        'Average Utilization Rate': df['utilization_rate'].mean(),
        'Max Utilization Rate': df['utilization_rate'].max(),
        'Average Running Jobs': df['running_jobs'].mean(),
        'Max Running Jobs': df['running_jobs'].max(),
        'Average Waiting Jobs': df['waiting_jobs'].mean(),
        'Max Waiting Jobs': df['waiting_jobs'].max(),
    }
    
    # Print statistics
    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # Calculate average node states
    node_states = {
        'Average Computing Nodes': df['nb_computing'].mean(),
        'Average Idle Nodes': df['nb_idle'].mean(),
        'Average Sleeping Nodes': df['nb_sleeping'].mean(),
        'Average Powered Off Nodes': df['nb_powered_off'].mean(),
    }
    
    print("\n=== Node States Statistics ===")
    for key, value in node_states.items():
        print(f"{key}: {value:.2f}")
    
    # Calculate state transitions
    transitions = {
        'Sleep Transitions': len(df[df['nb_switching_to_sleep'] > 0]),
        'Wake-up Transitions': len(df[df['nb_waking_from_sleep'] > 0]),
        'Power-on Transitions': len(df[df['nb_powering_on'] > 0]),
        'Power-off Transitions': len(df[df['nb_powering_off'] > 0]),
    }
    
    print("\n=== State Transitions Statistics ===")
    for key, value in transitions.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    file_path = "record.csv"
    analyze_power_control_record(file_path)