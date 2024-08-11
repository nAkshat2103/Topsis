import pandas as pd
import numpy as np
import sys
import argparse

def topsis(input_file, weights, impacts, output_file):
    # Read the input data
    data = pd.read_csv(input_file)
    weights = [float(w) for w in weights.split(',')]
    impacts = impacts.split(',')

    # Normalize the decision matrix
    norm_data = data.iloc[:, 1:].apply(lambda x: x / np.sqrt(np.sum(x**2)), axis=0)

    # Weighted normalized matrix
    weighted_data = norm_data * weights

    # Determine ideal and negative-ideal solutions
    ideal_solution = []
    negative_ideal_solution = []
    for i, impact in enumerate(impacts):
        if impact == '+':
            ideal_solution.append(weighted_data.iloc[:, i].max())
            negative_ideal_solution.append(weighted_data.iloc[:, i].min())
        else:
            ideal_solution.append(weighted_data.iloc[:, i].min())
            negative_ideal_solution.append(weighted_data.iloc[:, i].max())

    # Calculate the separation measures
    separation_pos = np.sqrt(np.sum((weighted_data - ideal_solution) ** 2, axis=1))
    separation_neg = np.sqrt(np.sum((weighted_data - negative_ideal_solution) ** 2, axis=1))

    # Calculate the relative closeness to the ideal solution
    scores = separation_neg / (separation_pos + separation_neg)

    # Rank the models
    data['TOPSIS Score'] = scores
    data['Rank'] = data['TOPSIS Score'].rank(ascending=False)

    # Save the results to the output file
    data.to_csv(output_file, index=False)

    print("TOPSIS analysis completed and saved to", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TOPSIS Implementation')
    parser.add_argument('input_file', help='Input CSV file containing the data')
    parser.add_argument('weights', help='Weights for the criteria')
    parser.add_argument('impacts', help='Impacts for the criteria (+/-)')
    parser.add_argument('output_file', help='Output CSV file to save the results')
    
    args = parser.parse_args()
    topsis(args.input_file, args.weights, args.impacts, args.output_file)
