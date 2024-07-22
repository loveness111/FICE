from new_plus import *

from neighbor_generator import *
from gpdatagenerator import calculate_feature_values


def newexplain_plus(idx_record2explain, X2E, dataset, blackbox, importance_df,
                    ng_function=genetic_neighborhood,
                    discrete_use_probabilities=False,
                    continuous_function_estimation=False,
                    returns_infos=False, path='./', sep=';', log=False):
    random.seed(0)
    np.random.seed(0)

    class_name = dataset['class_name']
    columns_tmp = dataset['columns_tmp']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']
    possible_outcomes = dataset['possible_outcomes']

    # Dataset Preprocessing
    dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous, 1000,
                                                         discrete_use_probabilities, continuous_function_estimation)

    dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)

    # Generate Neighborhood
    dfZ, Z = ng_function(dfZ, x, blackbox, dataset)
    rules = generate_rules(dfZ, class_name, columns, dataset, discrete, continuous, label_encoder, path=path, sep=sep,
                           log=log)
    print("rules:", rules)
    print("List of column names for training dt:", columns_tmp)
    print("List of column names for x:", dfZ.columns[1:])
    print("X data points:", x)
    dt = RuleBasedClassifier(rules, feature_names=columns_tmp)

    # Apply Black Box and Decision Tree on instance to explain
    cc_outcome, matched_rule = dt.predict([x])
    print("The prediction result of this method for x is:", cc_outcome)
    print("Rules matched by this method:", matched_rule)
    bb_outcome = blackbox.predict(x.reshape(1, -1))
    print("The black box model predicts x:", dataset['label_encoder'][dataset['class_name']].classes_[bb_outcome])
    # Apply Black Box and Decision Tree on neighborhood
    y_pred_bb = blackbox.predict(Z)
    leaf_nodes = None

    # Extract Coutnerfactuals
    diff_outcome = get_diff_outcome(cc_outcome, possible_outcomes)
    print("The categories to which counterfactuals should belong:", diff_outcome)
    counterfactuals = get_counterfactuals(dt, matched_rule, diff_outcome, x, importance_df, class_name, continuous, features_type)
    best_counterfactual = get_best_counterfactual(counterfactuals)
    explanation = (
        matched_rule, best_counterfactual, cc_outcome,
        dataset['label_encoder'][dataset['class_name']].classes_[bb_outcome])

    infos = {
        'bb_outcome': bb_outcome,
        'cc_outcome': cc_outcome,
        'y_pred_bb': y_pred_bb,
        'dfZ': dfZ,
        'Z': Z,
        'dt': dt,
        'leaf_nodes': leaf_nodes,
        'diff_outcome': diff_outcome
    }

    if returns_infos:
        return explanation, infos

    return explanation
