import lore
import pandas

import newexplain_plus
from new_plus import find_matching_data, get_class_counts
from prepare_dataset import *
from neighbor_generator import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from evaluation import evaluate_explanation

warnings.filterwarnings("ignore")


def main():
    path_data = 'datasets/'
    dataset_name = 'compas-scores-two-years.csv'
    dataset = prepare_compass_dataset(dataset_name, path_data)
    print(dataset['label_encoder'][dataset['class_name']].classes_)
    print(dataset['possible_outcomes'])
    feature_name = dataset['columns'][1:]

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    blackbox = RandomForestClassifier(n_estimators=20, random_state=0)
    blackbox.fit(X_train, y_train)
    importances = blackbox.feature_importances_
    if len(feature_name) != len(importances):
        raise ValueError(
            f"Lengths do not match: feature_names({len(feature_name)}) vs importances({len(importances)})")
    importance_df = pd.DataFrame({'feature': feature_name, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    # print("Feature Importance Ranking:", importance_df)

    X2E = X_test
    '''
    y2E = blackbox.predict(X2E)
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])
    '''
    idx_record2explain = 0

    explanation, infos = newexplain_plus.newexplain_plus(idx_record2explain, X2E, dataset, blackbox, importance_df,
                                                         ng_function=genetic_neighborhood,
                                                         discrete_use_probabilities=True,
                                                         continuous_function_estimation=False,
                                                         returns_infos=True,
                                                         path=path_data, sep=';', log=False)

    dfX2E = build_df2explain(blackbox, X2E, dataset).to_dict('records')
    dfx = dfX2E[idx_record2explain]
    # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]

    print('The eigenvalue of the x data point = %s' % dfx)
    matched_rule, best_counterfactual, cc_outcome, class_label = explanation
    print("x matches the rule = ", matched_rule)
    print("The prediction result of this method for x is = ", cc_outcome)
    print("The black box model’s prediction for x = ", class_label)
    print("The nearest counterfactual rule for x:", best_counterfactual)

    matching_data = find_matching_data(dfX2E, matched_rule)
    print("The number of data points in the test set that are covered by the rule x:", len(matching_data))
    class_count = get_class_counts(matching_data)
    for class_value, count in class_count.items():
        print(f"category:{class_value}, The number of occurrences:{count}")
    print("The accuracy of this method in simulating the prediction results of the black box model (random forest):",
          (class_count[cc_outcome] / sum(class_count.values())))


if __name__ == "__main__":
    main()
