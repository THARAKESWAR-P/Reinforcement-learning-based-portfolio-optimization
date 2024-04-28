import re
import matplotlib.pyplot as plt

# Initialize lists to store iteration numbers and average returns
iterations = []
avg_returns = []

# Read the log file
with open('training_03.log', 'r') as file:
    # Iterate through each line in the file
    for line in file:
        # Use regular expressions to extract iteration number and average return
        match = re.search(r'step = (\d+).*Average Return = ([\d\.-]+)', line)
        if match:
            iteration = int(match.group(1))
            avg_return = float(match.group(2))
            # Append to lists
            iterations.append(iteration)
            avg_returns.append(avg_return)

# Plot the data
plt.plot(iterations, avg_returns)
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.title('Average Return vs Iteration')
plt.grid(True)
plt.show()
