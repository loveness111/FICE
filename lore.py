import pyyadt
import random
from new_plus import *

from neighbor_generator import *
from gpdatagenerator import calculate_feature_values
from sklearn.neighbors import KNeighborsClassifier


def explain(idx_record2explain, X2E, dataset, blackbox,
            ng_function=genetic_neighborhood,  # generate_random_data, #genetic_neighborhood, random_neighborhood
            discrete_use_probabilities=False,
            continuous_function_estimation=False,
            returns_infos=False, path='./', sep=';', log=False):

    random.seed(0)
    np.random.seed(0)

    class_name = dataset['class_name']
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

    # Build Decision Tree
    dt, dt_dot = pyyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous,
                            filename=dataset['name'], path=path, sep=sep, log=log)

    # Apply Black Box and Decision Tree on instance to explain
    print("x data points:", x)
    bb_outcome = blackbox.predict(x.reshape(1, -1))

    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]

    cc_outcome, rule, tree_path = pyyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)

    # Apply Black Box and Decision Tree on neighborhood
    y_pred_bb = blackbox.predict(Z)
    y_pred_cc, leaf_nodes = pyyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type, discrete, continuous)

    def predict(X):
        y, ln, = pyyadt.predict(dt, X, class_name, features_type, discrete, continuous)
        return y, ln

    # Update labels if necessary
    if class_name in label_encoder:
        cc_outcome = label_encoder[class_name].transform(np.array([cc_outcome]))[0]

    if class_name in label_encoder:
        y_pred_cc = label_encoder[class_name].transform(y_pred_cc)

    # Extract Coutnerfactuals
    diff_outcome = get_diff_outcome(cc_outcome,
                                    possible_outcomes)
    counterfactuals = pyyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome,
                                                 class_name, continuous, features_type)

    explanation = (rule, counterfactuals)

    infos = {
        'bb_outcome': bb_outcome,
        'cc_outcome': cc_outcome,
        'y_pred_bb': y_pred_bb,
        'y_pred_cc': y_pred_cc,
        'dfZ': dfZ,
        'Z': Z,
        'dt': dt,
        'tree_path': tree_path,
        'leaf_nodes': leaf_nodes,
        'diff_outcome': diff_outcome,
        'predict': predict,
    }

    if returns_infos:
        return explanation, infos

    return explanation


def newexplain(idx_record2explain, X2E, dataset, blackbox,
               ng_function=genetic_neighborhood,  # generate_random_data, #genetic_neighborhood, random_neighborhood
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

    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
    print("LabelEncoder.classes:", label_encoder['class'].classes_)
    print("LabelEncoder:", label_encoder)
    cc_outcome, matched_rule = dt.predict([x])
    print("The prediction result of this method for x is:", cc_outcome)
    print("Rules matched by this method:", matched_rule)
    bb_outcome = blackbox.predict(x.reshape(1, -1))
    print("The black box model's prediction for x:", dataset['label_encoder'][dataset['class_name']].classes_[bb_outcome])
    # Apply Black Box and Decision Tree on neighborhood
    y_pred_bb = blackbox.predict(Z)
    leaf_nodes = None

    # Extract Coutnerfactuals
    diff_outcome = get_diff_outcome(cc_outcome,  possible_outcomes)
    print("The category to which the counterfactual should belong:", diff_outcome)
    counterfactuals = get_counterfactuals(dt, matched_rule, diff_outcome, class_name, continuous, features_type)
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


def is_satisfied(x, rule, discrete, features_type):
    for col, val in rule.items():
        if col in discrete:
            if str(x[col]).strip() != val:
                return False
        else:
            if '<=' in val and '<' in val and val.find('<=') < val.find('<'):
                val = val.split(col)
                thr1 = pyyadt.yadt_value2type(val[0].replace('<=', ''), col, features_type)
                thr2 = pyyadt.yadt_value2type(val[1].replace('<', ''), col, features_type)
                if x[col] > thr1 or x[col] <= thr2:
                    return False
            elif '<' in val and '<=' in val and val.find('<') < val.find('<='):
                val = val.split(col)
                thr1 = pyyadt.yadt_value2type(val[0].replace('<', ''), col, features_type)
                thr2 = pyyadt.yadt_value2type(val[1].replace('<=', ''), col, features_type)
                if x[col] >= thr1 or x[col] < thr2:
                    return False
            elif '<=' in val:
                thr = pyyadt.yadt_value2type(val.replace('<=', ''), col, features_type)
                if x[col] > thr:
                    return False
            elif '>' in val:
                thr = pyyadt.yadt_value2type(val.replace('>', ''), col, features_type)
                if x[col] <= thr:
                    return False
    return True


def get_covered(rule, X, dataset):
    covered_indexes = list()
    for i, x in enumerate(X):
        if is_satisfied(x, rule, dataset['discrete'], dataset['features_type']):
            covered_indexes.append(i)
    return covered_indexes
