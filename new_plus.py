# rule_based_classifier.py
import os
import subprocess
from collections import Counter

import pandas as pd
import networkx as nx
import numpy as np
import re

import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder

import pyyadt
from util import label_decode


class RuleBasedClassifier:
    def __init__(self, rules, feature_names, default_prediction=None):
        self.rules = self._parse_rules(rules)
        print("The rules for the current classifier object recognition:")
        for rule in self.rules:
            print(rule)
        self.rules = self._merge_age_and_age_cat()
        print("The new rule after the current classifier object merges the age and age_cat columns:")
        for rule in self.rules:
            print(rule)
        self.feature_names = feature_names
        self.default_prediction = default_prediction

    def _merge_age_and_age_cat(self):
        merged_rules = []
        for rule in self.rules:
            if not any(cond[0] == 'age' for cond in rule) or not any(cond[0] == 'age_cat' for cond in rule):
                merged_rules.append(rule)
                continue
            age_conditions = [cond for cond in rule if cond[0] == 'age']
            print("Currently obtained age_conditions:", age_conditions)
            age_cat_conditions = [cond for cond in rule if cond[0] == 'age_cat']
            print("Currently obtained age_cat_conditions:", age_cat_conditions)
            if age_conditions and age_cat_conditions:
                # Combine age and age_cat conditions
                age_operator, age_value = age_conditions[0][1], age_conditions[0][2]
                age_cat_operator, age_cat_value = age_cat_conditions[0][1], age_cat_conditions[0][2]
                if age_operator == 'between' and age_cat_operator == 'between':
                    age_cat_low, age_cat_high = map(int, age_cat_value)
                    age_low, age_high = map(int, age_value)
                    if age_cat_low <= age_low <= age_high <= age_cat_high:
                        new_age_value = (float(age_low), float(age_high))
                        new_age_condition = ('age', 'between', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                    elif age_low > age_cat_high or age_high < age_cat_low:
                        continue
                    elif age_low <= age_cat_low <= age_cat_high <= age_high:
                        new_age_value = (float(age_cat_low), float(age_cat_high))
                        new_age_condition = ('age', 'between', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                    elif age_low <= age_cat_low <= age_high <= age_cat_high:
                        new_age_value = (float(age_cat_low), float(age_high))
                        new_age_condition = ('age', 'between', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                    else:
                        new_age_value = (float(age_low), float(age_cat_high))
                        new_age_condition = ('age', 'between', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                elif age_operator == '<=' and age_cat_operator == 'between':
                    age_value = int(age_value)
                    age_cat_low, age_cat_high = map(int, age_cat_value)
                    if age_cat_low < age_value < age_cat_high:
                        new_age_value = (float(age_cat_low), float(age_value))
                        new_age_condition = ('age', 'between', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                    elif age_cat_low < age_cat_high < age_value:
                        new_age_value = age_value
                        new_age_condition = ('age', '<=', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                    else:
                        continue
                elif age_operator == '<=' and age_cat_operator == 'Greater than':
                    age_value = int(age_value)
                    age_cat_value = int(age_cat_value)
                    if age_cat_value < age_value:
                        new_age_value = (float(age_cat_value), float(age_value))
                        new_age_condition = ('age', 'between', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                    else:
                        continue
                elif age_operator == '<=' and age_cat_operator == 'Less than':
                    age_value = int(age_value)
                    age_cat_value = int(age_cat_value)
                    if age_cat_value < age_value:
                        new_age_value = age_cat_value
                        new_age_condition = ('age', '<=', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                    else:
                        new_age_value = age_value
                        new_age_condition = ('age', '<=', new_age_value)
                        rule = [cond for cond in rule if cond[0] != 'age_cat']
                        rule = [cond for cond in rule if cond[0] != 'age']
                        merged_rules.append(rule[:-1] + [new_age_condition] + [rule[-1]])
                else:  # Delete incorrect rules
                    continue
        return merged_rules

    def _parse_rules(self, rules):
        parsed_rules = []
        for rule in rules:
            print("The rules currently recognized by _parse_rules are:", rule)

            # Split the rule into conditions and prediction
            conditions_part, prediction_part = rule.split(' => ')
            print("Currently identified conditions and predicted results:", conditions_part, prediction_part)
            prediction = prediction_part.split('(')[0].strip()

            # Split conditions by ' AND '
            conditions = conditions_part.split(' AND ')
            print("All conditions currently recognized:", conditions)
            parsed_rule = []

            # Parse each condition
            for condition in conditions:
                print("Features currently being processed:", condition)
                if '<=' in condition:
                    feature, value = condition.split('<=')
                    operator = '<='
                elif '>=' in condition:
                    feature, value = condition.split('>=')
                    operator = '>='
                elif '<' in condition:
                    feature, value = condition.split('<')
                    operator = '<'
                elif '>' in condition:
                    feature, value = condition.split('>')
                    operator = '>'
                elif '=' in condition:
                    feature, value = condition.split('=')
                    operator = '='
                else:
                    parts = condition.split()
                    if len(parts) == 4:
                        if parts[1] == 'Greater':
                            feature = parts[0]
                            operator = 'Greater than'
                            value = parts[3]
                        elif parts[1] == 'Less':
                            feature = parts[0]
                            operator = 'Less than'
                            value = parts[3]
                        else:
                            feature = parts[0]
                            operator = '25 - 45'
                            value = None
                    else:
                        feature = parts[0]
                        operator = '='
                        value = parts[1]

                feature = feature.strip()
                operator = operator.strip()
                value = value.strip() if value is not None else value
                print("Current features:", feature, "Current operator：", operator, "Current value：", value)
                feature_exists = False
                for i, (parsed_feature, parsed_operator, parsed_value) in enumerate(parsed_rule):
                    if parsed_feature == feature:
                        # Feature already exists, merge the conditions
                        parsed_value = parsed_value.strip() if parsed_value is not None else parsed_value
                        if parsed_operator == '<=' and operator == '>=':
                            low, high = min(parsed_value, value), max(parsed_value, value)
                            parsed_rule[i] = (feature, 'between', (low, high))
                        elif parsed_operator == '>=' and operator == '<=':
                            low, high = min(value, parsed_value), max(value, parsed_value)
                            parsed_rule[i] = (feature, 'between', (low, high))
                        elif parsed_operator == '<=' and operator == '<=':
                            parsed_rule[i] = (feature, '<=', min(parsed_value, value))
                        elif parsed_operator == '>=' and operator == '>=':
                            parsed_rule[i] = (feature, '>=', max(parsed_value, value))
                        elif parsed_operator == '<=' and operator == '>':
                            low, high = min(parsed_value, value), max(parsed_value, value)
                            parsed_rule[i] = (feature, 'between', (low, high))
                        elif parsed_operator == '>' and operator == '<=':
                            low, high = min(value, parsed_value), max(value, parsed_value)
                            parsed_rule[i] = (feature, 'between', (low, high))
                        elif parsed_operator == '>' and operator == '>=':
                            parsed_rule[i] = (feature, '>=', max(parsed_value, value))
                        elif parsed_operator == '>=' and operator == '>':
                            parsed_rule[i] = (feature, '>=', max(parsed_value, value))
                        elif parsed_operator == '<=' and operator == '<':
                            parsed_rule[i] = (feature, '<=', min(parsed_value, value))
                        elif parsed_operator == '<' and operator == '<=':
                            parsed_rule[i] = (feature, '<=', min(parsed_value, value))
                        elif parsed_operator == '>' and operator == '<':
                            low, high = min(parsed_value, value), max(parsed_value, value)
                            parsed_rule[i] = (feature, 'between', (low, high))
                        elif parsed_operator == '<' and operator == '>':
                            low, high = min(value, parsed_value), max(value, parsed_value)
                            parsed_rule[i] = (feature, 'between', (low, high))
                        elif parsed_operator == '=' and operator == '=':
                            parsed_rule[i] = (feature, '=', parsed_value)  # Both are equal, keep one
                        elif parsed_operator == '=':
                            parsed_rule[i] = (feature, operator, value)
                        elif operator == '=':
                            pass  # Do nothing, keep the existing condition
                        feature_exists = True
                        break

                if not feature_exists:
                    # If the feature is not already in the parsed rule, add it
                    parsed_rule.append((feature, operator, value))
            # Process age_cat rules and convert to 'between' format
            for i, (parsed_feature, parsed_operator, parsed_value) in enumerate(parsed_rule):
                if parsed_operator == '25 - 45':
                    parsed_rule[i] = (parsed_feature, 'between', ('25', '45'))

            parsed_rule.append(prediction)
            parsed_rules.append(parsed_rule)

        return parsed_rules

    def _satisfies_rule(self, sample, rule):
        for feature, operator, value in rule[:-1]:
            feature_index = self.feature_names.index(feature)
            sample_value = sample[feature_index]
            if operator == '<=' and not (sample_value <= float(value)):
                return False
            if operator == '>=' and not (sample_value >= float(value)):
                return False
            if operator == '<' and not (sample_value < float(value)):
                return False
            if operator == '>' and not (sample_value > float(value)):
                return False
            if operator == '=' and not (sample_value == float(value) or sample_value == value):
                return False
            if operator == '-' and not (float(value.split()[0]) <= sample_value <= float(value.split()[2])):
                return False
        return True

    def predict(self, X):
        if isinstance(X, dict):
            samples = [X]
        else:
            samples = X

        predictions = []
        matched_rules = []
        for sample in samples:
            print("Current sample：", sample)
            matched = False
            for rule in self.rules:
                if self._satisfies_rule(sample, rule):
                    predictions.append(rule[-1])
                    matched_rules.append(rule)
                    matched = True
                    break
            if not matched:
                predictions.append(None)  # or some default value
                matched_rules.append(None)  # No rule matched

        if len(predictions) == 1:
            return predictions[0], matched_rules[0]
        return predictions, matched_rules


def match_condition(value, operator, rule_value):
    """ Helper function to match a value against a rule's condition """
    if operator == '=':
        return value == rule_value
    elif operator == '<=':
        return value <= rule_value
    elif operator == '>=':
        return value >= rule_value
    elif operator == '<':
        return value < rule_value
    elif operator == '>':
        return value > rule_value
    elif operator == 'between':
        if isinstance(rule_value, tuple):
            low, high = rule_value
            if isinstance(value, tuple):  # If value is also a tuple
                return low <= value[0] and value[1] <= high
            else:  # If value is a single number
                return low < value < high
        else:
            False
    return False


def parse_value_(value):
    """ Helper function to parse value into float or tuple of floats """
    if isinstance(value, tuple):
        return tuple(map(float, value))
    if isinstance(value, str) and ' - ' in value:
        low, high = map(float, value.split(' - '))
        return (low, high)
    try:
        return float(value)
    except ValueError:
        return value


def find_matching_data(dfX2E, rule):
    matching_data = []
    for data in dfX2E:
        match = True
        for feature, operator, rule_value in rule[:-1]:  # Exclude the class label
            data_value = data.get(feature.strip())
            if data_value is None:
                match = False
                break
            if isinstance(data_value, str):
                if "Less than" in data_value:
                    data_value = data_value.replace("Less than ", "")
                elif "Greater than" in data_value:
                    data_value = data_value.replace("Greater than ", "")
            parsed_rule_value = parse_value_(rule_value)
            parsed_data_value = parse_value_(data_value)

            # if isinstance(parsed_rule_value, tuple) and not isinstance(parsed_data_value, tuple):
            #   parsed_data_value = (parsed_data_value, parsed_data_value)
            # if isinstance(parsed_data_value, tuple) and not isinstance(parsed_rule_value, tuple):
            #   parsed_rule_value = (parsed_rule_value, parsed_rule_value)

            if not match_condition(parsed_data_value, operator, parsed_rule_value):
                match = False
                break
        if match:
            matching_data.append(data)
    return matching_data


def get_class_counts(matching_data, class_name='class'):
    """
    Function to count the occurrences of each class in the matching data.

    :param matching_data: List of dictionaries, each containing data including class value.
    :param class_name: The key name used to identify the class value in the dictionaries.
    :return: A dictionary with two class counts.
    """
    class_values = [data[class_name] for data in matching_data if class_name in data]
    class_counts = Counter(class_values)

    # Ensure we have exactly two classes in the result
    if len(class_counts) != 2:
        raise ValueError("The data does not contain exactly two unique class values.")

    return dict(class_counts)


def _satisfies_rule(self, sample, rule):
    for condition in rule[:-1]:
        feature, operator, value = condition
        feature_index = self.feature_names.index(feature)
        sample_value = sample[feature_index]
        if operator == '<=':
            if not sample_value <= float(value):
                return False
        elif operator == '>=':
            if not sample_value >= float(value):
                return False
        elif operator == '<':
            if not sample_value < float(value):
                return False
        elif operator == '>':
            if not sample_value > float(value):
                return False
        elif operator == '=':
            if not str(sample_value) == value:
                return False
    return True


def predict(self, sample):
    prediction = self.default_prediction
    for rule in self.rules:
        if self._satisfies_rule(sample, rule):
            prediction = rule[-1]
            break
    return prediction


def get_rule(tree_path, class_name, y, node_labels=None, edge_labels=None, dt=None):
    if node_labels is None:
        node_labels = get_node_labels(dt)

    if edge_labels is None:
        edge_labels = get_edge_labels(dt)

    ant = dict()
    for i in range(0, len(tree_path) - 1):
        node = tree_path[i]
        child = tree_path[i + 1]
        if (node, child) in edge_labels:
            att = node_labels[node]
            val = edge_labels[(node, child)]
        else:
            att = node_labels[child]
            val = edge_labels[(child, node)]

        if att in ant:
            val0 = ant[att]
            min_thr0 = None
            max_thr0 = None

            min_thr = None
            max_thr = None

            if len(re.findall('.*<.*<=.*', val0)):
                min_thr0 = float(val0.split('<')[0])
                max_thr0 = float(val0.split('<=')[1])
            elif '<=' in val0:
                max_thr0 = float(val0.split('<=')[1])
            elif '>' in val0:
                min_thr0 = float(val0.split('>')[1])

            if len(re.findall('.*<.*<=.*', val)):
                min_thr = float(val.split('<')[0])
                max_thr = float(val.split('<=')[1])
            elif '<=' in val:
                max_thr = float(val.split('<=')[1])
            elif '>' in val:
                min_thr = float(val.split('>')[1])

            new_min_thr = None
            new_max_thr = None

            if min_thr:
                new_min_thr = max(min_thr, min_thr0) if min_thr0 else min_thr

            if min_thr0:
                new_min_thr = max(min_thr, min_thr0) if min_thr else min_thr0

            if max_thr:
                new_max_thr = min(max_thr, max_thr0) if max_thr0 else max_thr

            if max_thr0:
                new_max_thr = min(max_thr, max_thr0) if max_thr else max_thr0

            if new_min_thr and new_max_thr:
                val = '%s< %s <=%s' % (new_min_thr, att, new_max_thr)
            elif new_min_thr:
                val = '>%s' % new_min_thr
            elif new_max_thr:
                val = '<=%s' % new_max_thr

        ant[att] = val

    cons = {class_name: y}

    weights = node_labels[tree_path[-1]].split('(')[1]
    weights = weights.replace(')', '')
    weights = [float(w) for w in weights.split('/')]

    rule = [cons, ant, weights]

    return rule


def fit(df, class_name, columns, features_type, discrete, continuous,
        filename='yadt_dataset', path='./', sep=';', log=False):
    data_filename = path + filename + '.data'
    names_filename = path + filename + '.names'
    tree_filename = path + filename + '.dot'

    df.to_csv(data_filename, sep=sep, header=False, index=False)

    names_file = open(names_filename, 'w')
    for col in columns:
        col_type = features_type[col]
        disc_cont = 'discrete' if col in discrete else 'continuous'
        disc_cont = 'class' if col == class_name else disc_cont
        names_file.write('%s%s%s%s%s\n' % (col, sep, col_type, sep, disc_cont))
    names_file.close()

    cmd = 'yadt/dTcmd -fd %s -fm %s -sep %s -d %s' % (
        data_filename, names_filename, sep, tree_filename)
    output = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)

    if log:
        print(cmd)
        print(output)

    dt = nx.DiGraph(nx.drawing.nx_pydot.read_dot(tree_filename))
    dt_dot = pydotplus.graph_from_dot_data(open(tree_filename, 'r').read())

    if os.path.exists(data_filename):
        os.remove(data_filename)

    if os.path.exists(names_filename):
        os.remove(names_filename)

    return dt, dt_dot


def generate_rules(df, class_name, columns, datasets, discrete, continuous, label_encoder, path, sep, log):
    X = df.drop(class_name, axis=1)
    y = df[class_name]
    features_name = datasets['columns']
    features_type = datasets['features_type']

    for col in discrete:
        if col != class_name:
            le = label_encoder[col]
            X[col] = le.fit_transform(X[col])

    if class_name in discrete:
        le = label_encoder[col]
        y = le.fit_transform(y)

    dt, dt_dot = pyyadt.fit(df, class_name, columns, features_type, discrete, continuous,
                            filename=datasets['name'], path=path, sep=sep, log=log)

    rules = extract_rules_from_dot(dt_dot)
    print("All rules extracted:")
    for rule in rules:
        print(rule)
    return rules


def extract_rules_from_dot(graph):
    rules = []

    def recurse(node, path):
        node_label = node.get('label').strip('"').replace("\\n", "")
        print(f"Processing node: {node.get_name()}, label: {node_label}")  # Debug information

        if "High" in node_label or "Medium-Low" in node_label:
            rule = " AND ".join(path) + " => " + node_label
            rules.append(rule)
        else:
            for edge in graph.get_edges():
                if edge.get_source() == node.get_name():
                    label = edge.get_label().strip('"')
                    next_node = graph.get_node(edge.get_destination())[0]
                    recurse(next_node, path + [f"{node_label} {label}"])

    root = graph.get_node('n0')[0]
    recurse(root, [])

    print("Final rules:")  # Debug information
    for rule in rules:
        print(rule)

    return rules


def get_rules_as_graph(dt):
    G = nx.DiGraph()
    for i, rule in enumerate(dt.rules):
        conditions = " AND ".join([f"{feature} {operator} {value}" for feature, operator, value in rule[:-1]])
        prediction = rule[-1]
        node_label = f"{conditions} => {prediction}"
        G.add_node(i, label=node_label)
        if i > 0:
            G.add_edge(i - 1, i, label=conditions)
    return G


def get_edge_labels(dt):
    return {edge: label.replace('"', '').replace('\\n', '') for edge, label in dt.edges(data='label')}

def get_node_labels(dt):
    return {node: label.replace('"', '').replace('\\n', '') for node, label in dt.nodes(data='label')}


def expand_rule(rule, continuous):
    # Assuming expand_rule is a function to process rule expansion based on continuous features
    expanded_rule = {}
    conditions = rule.split(' AND ')
    for condition in conditions:
        feature, operator, value = condition.split(' ')
        expanded_rule[feature] = (operator, value)
    return expanded_rule


def yadt_value2type(value, class_name, features_type):
    # Assuming yadt_value2type is a function to convert value based on class_name and features_type
    return value


def get_falsifeid_conditions1(cond, ccond, continuous):
    # Assuming get_falsifeid_conditions1 is a function to get falsified conditions between two rules
    delta = {k: v for k, v in ccond.items() if k not in cond or cond[k] != v}
    qlen = len(delta)
    return delta, qlen


def get_rule(node_path, class_name, diff_outcome, node_labels, edge_labels):
    # Assuming get_rule is a function to construct rule from node path
    rule = ""
    for node in node_path:
        rule += node_labels[node] + " AND "
    return rule[:-5]  # Remove the last ' AND '


def calculate_diff(matched_rule, rule2, x, importance_df):
    print("The rule that is calculating the difference with x:", rule2)

    def parse_value(value):
        """Parse value to handle tuples or individual values uniformly."""
        if isinstance(value, tuple):
            return tuple(map(float, value))
        try:
            return float(value)
        except ValueError:
            return value

    def compare_values(val1, val2, operator1, operator2):
        """Compare values considering operators."""
        if operator1 == 'between':
            low1, high1 = val1
            if operator2 == '<=':
                return low1 <= val2 <= high1
            if operator2 == '>=':
                return low1 <= val2 <= high1
            if operator2 == '<':
                return low1 < val2 <= high1
            if operator2 == '>':
                return low1 <= val2 < high1
            if operator2 == 'between':
                low2, high2 = val2
                return low1 <= high2 and low2 <= high1

        if operator2 == 'between':
            low2, high2 = val2
            if operator1 == '<=':
                return low2 <= val1 <= high2
            if operator1 == '>=':
                return low2 <= val1 <= high2
            if operator1 == '<':
                return low2 < val1 <= high2
            if operator1 == '>':
                return low2 <= val1 < high2

        # Handle other combinations of operators
        if operator1 == '<=':
            if operator2 == '<=':
                return val1 == val2
            if operator2 == '>=':
                return val2 <= val1
            if operator2 == '<':
                return val1 <= val2
            if operator2 == '>':
                return val2 > val1

        if operator1 == '>=':
            if operator2 == '<=':
                return val1 <= val2
            if operator2 == '>=':
                return val1 == val2
            if operator2 == '<':
                return val1 < val2
            if operator2 == '>':
                return val1 >= val2

        if operator1 == '<':
            if operator2 == '<=':
                return val1 <= val2
            if operator2 == '>=':
                return val2 <= val1
            if operator2 == '<':
                return val1 == val2
            if operator2 == '>':
                return False  # No overlap possible

        if operator1 == '>':
            if operator2 == '<=':
                return val1 < val2
            if operator2 == '>=':
                return val1 >= val2
            if operator2 == '<':
                return False  # No overlap possible
            if operator2 == '>':
                return val1 == val2

        return val1 == val2

    def compare_values_for_priors_count(x_priors_count, parsed_value2, operator2):
        """Compare x_priors_count against parsed_value2 with operator2."""
        if operator2 == 'between':
            if isinstance(parsed_value2, tuple):
                low2, high2 = parsed_value2
            else:
                low2, high2 = parsed_value2, parsed_value2
            return low2 < x_priors_count < high2

        if operator2 == '<=':
            return x_priors_count <= parsed_value2
        if operator2 == '>=':
            return x_priors_count >= parsed_value2
        if operator2 == '<':
            return x_priors_count < parsed_value2
        if operator2 == '>':
            return x_priors_count > parsed_value2

        return x_priors_count == parsed_value2

    common_features = []
    matched_features = {f1.strip() for f1, _, _ in matched_rule[:-1]}
    rule2_features = {f2.strip() for f2, _, _ in rule2[:-1]}

    common_feature_names = matched_features.intersection(rule2_features)
    for feature in common_feature_names:
        importance = importance_df.loc[importance_df['feature'] == feature, 'importance'].values[0]
        common_features.append((feature, importance))
    print("Common features and feature importance:", common_features)
    total_importance = sum(importance for _, importance in common_features)

    normalized_common_features = [(feature, importance / total_importance) for feature, importance in common_features]
    print("Common features and feature importance after normalization:", normalized_common_features)

    diff_features = []

    for feature1, operator1, value1 in matched_rule[:-1]:
        for feature2, operator2, value2 in rule2[:-1]:
            if feature1.strip() == feature2.strip():
                if feature1.strip() == 'priors_count':
                    x_priors_count = x[4]
                    print("The priors_count value of x:", x_priors_count)
                    if x_priors_count is not None:
                        x_priors_count = parse_value(x_priors_count)
                        parsed_value2 = parse_value(value2)
                        if not compare_values_for_priors_count(x_priors_count, parsed_value2, operator2):
                            importance = next(importance for feat, importance in normalized_common_features if
                                              feat == feature1.strip())
                            diff_features.append((feature1.strip(), importance))
                elif feature1.strip() == 'age':
                    x_age = x[0]
                    print("The age value of x:", x_age)
                    if x_age is not None:
                        x_age = parse_value(x_age)
                        parsed_value2 = parse_value(value2)
                        if not compare_values_for_priors_count(x_age, parsed_value2, operator2):
                            importance = next(importance for feat, importance in normalized_common_features if
                                              feat == feature1.strip())
                            diff_features.append((feature1.strip(), importance))
                else:
                    parsed_value1 = parse_value(value1)
                    parsed_value2 = parse_value(value2)
                    if not compare_values(parsed_value1, parsed_value2, operator1, operator2):
                        importance = next(
                            importance for feat, importance in normalized_common_features if feat == feature1.strip())
                        diff_features.append((feature1.strip(), importance))
    print("Different features and feature importance:", diff_features)
    return diff_features, normalized_common_features


def get_counterfactuals(dt, matched_rule, diff_outcome, x, importance_df, class_name, continuous,
                        features_type):
    counterfactuals = []
    for rule in dt.rules:
        print("The current rule is:", rule)
        if rule[-1] != diff_outcome:
            continue
        diff_features, common_features = calculate_diff(matched_rule, rule, x, importance_df)
        counterfactuals.append((rule, diff_features, common_features))
    print("Here are all the counterfactual rules found:")
    for idx, (rule, diff_features, common_features) in enumerate(counterfactuals, start=1):
        print(f"Counterfactual Rules{idx}：{rule},Differentiating features:{diff_features},Common feature:{common_features}")
    return counterfactuals


def get_best_counterfactual(counterfactuals):
    best_rule = None
    min_ratio = float('inf')

    for rule, diff_features, common_features in counterfactuals:
        total_diff_importance = sum(importance for _, importance in diff_features)
        total_common_importance = sum(importance for _, importance in common_features)

        if total_common_importance > 0:
            ratio = total_diff_importance / total_common_importance
            if ratio < min_ratio:
                min_ratio = ratio
                if best_rule is not None:
                    best_rule.clear()
                best_rule = rule
            elif ratio == min_ratio:
                best_rule.append(rule)

    return best_rule


def _expand_rule(rule, continuous):
    ant = rule[1]
    exp_ant = dict()
    for att in ant:
        if continuous[att]:
            if '<=' in ant[att]:
                min_thr = float(ant[att].split('<')[0])
                max_thr = float(ant[att].split('<=')[1])
                exp_ant[att] = '%s< %s <=%s' % (min_thr, att, max_thr)
            elif '>' in ant[att]:
                min_thr = float(ant[att].split('>')[1])
                exp_ant[att] = '>%s' % min_thr
        else:
            exp_ant[att] = ant[att]
    return exp_ant


def _yadt_value2type(value, class_name, features_type):
    if class_name in features_type:
        if features_type[class_name] == 'categorical':
            return value
        else:
            return float(value)
    else:
        return value


def zhuanhuanx(dfZ, discrete, label_encoder):
    dfZx = dfZ
    df_de = label_decode(dfZx, discrete, label_encoder)
    print("df_le:", df_de)
    x = df_de.iloc[0]
    df = pd.DataFrame([x])
    df = df.drop('class', axis=1)
    x = df.iloc[0].values
    return x
