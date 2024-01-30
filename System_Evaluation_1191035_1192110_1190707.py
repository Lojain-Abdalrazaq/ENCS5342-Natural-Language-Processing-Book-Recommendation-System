# Books Recommendation System Using Collaborative Filtering
# Nour Rabee' 1191035
# Najwa Bsharat 1192110
# Lojain Abdalrazaq 1190707
# NOTE:The project was run using JUPYTER NOTEBOOK 
# NOTE: The script represents the EVALUATION PHASE - PRECISION, RECALL, AVG PRECISION, PR-CURVE

import numpy as np

# Function to evaluate recommendations based on ground truth
def evaluate_recommendations(recommended_books, ground_truth):
    num_relevant_books = sum(ground_truth)
    precision_scores = []
    recall_scores = []
    average_precision_scores = []
    relevant_books_list = []

    relevant_count = 0
    for i, book in enumerate(recommended_books):
        if ground_truth[i] == 1:
            relevant_count += 1

        precision = relevant_count / (i + 1)
        recall = relevant_count / num_relevant_books
        average_precision = relevant_count / (i + 1) if ground_truth[i] == 1 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        average_precision_scores.append(average_precision)

        relevant_books = [book for j, book in enumerate(recommended_books[:i + 1]) if ground_truth[j] == 1]
        relevant_books_list.append(relevant_books)

    return precision_scores, recall_scores, average_precision_scores, relevant_books_list


recommended_books = ['62291', '157993', '22034', '2318271', '4381', '119322', '2767793', '78983', '119324', '13497']
ground_truth = [1, 0, 1, 1, 1, 0, 1, 1, 0, 1]
# Evaluate the recommended books
# precision, recall, average_precision
precision, recall, average_precision, relevant_books_list = evaluate_recommendations(recommended_books, ground_truth)
Actual_avg_prececion = 0
count = 0
# to find the Actual_avg_prececion value
for i in range(len(recommended_books)):
    if average_precision[i] > 0:
        count += 1
        Actual_avg_prececion += average_precision[i]

Actual_avg_prececion = Actual_avg_prececion / count
# printing the results
print("Precision:", precision)
print("Recall:", recall)
print("Precistion Values at the relevant books:", average_precision)
print("Average Precision", Actual_avg_prececion)
print("Relevant Books List:", relevant_books_list)
average_precision

import numpy as np
import matplotlib.pyplot as plt

# intialize interpolated precision array
interpolated_precision_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
interpolated_precision_y = [1, 1, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.7, 0.7]
recall_x = [0.142857143, 0.285714286, 0.428571429, 0.571428571, 0.714285714, 0.857142857, 1]
precision_y = [1.0, 0.666667, 0.75, 0.666667, 0.714285714, 0.75, 0.7]
# Plot interpolated P-R curve
plt.plot(recall_x, precision_y, '-o')
plt.plot(interpolated_precision_x, interpolated_precision_y, '-o')
plt.xlabel('recall_x')
plt.ylabel('precision_y')
plt.legend(['P-R Curve', 'Interpolated P-R Curve'])
plt.title('Precision-Recall Curve')
plt.show()