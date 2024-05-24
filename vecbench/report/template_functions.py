# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

template_functions = []


def template_function(func):
    template_functions.append({"name": func.__name__, "ptr": func})
    return func


@template_function
def unique_column(df, colname):
    return df[colname].unique()


@template_function
def max_column(df, colname):
    return df[colname].max()


@template_function
def quantile_field_column(df, field_name, quantile):
    return df[df["fields"] == field_name]["values"].quantile(quantile) * 1000


@template_function
def sum_field_column(df, field_name):
    return df[df["fields"] == field_name]["values"].sum()


@template_function
def sum_max_group_column(df, field_name, gby_col):
    gk = df[df["fields"] == field_name].groupby(gby_col)
    return gk["values"].max().sum()

@template_function
def flatten(dic, columns=[], values=[]):
    for key in dic.keys():
        if type(dic[key]) is str:
            columns.append(key)
            values.append(dic[key])
        if type(dic[key]) is dict:
            flatten(dic[key], columns, values)
    return columns, values


@template_function
def datetime_diff(df, field_name):
    return (
        df[df["fields"] == field_name]["timestamp"].max()
        - df[df["fields"] == field_name]["timestamp"].min()
    ).total_seconds()


def config_template(template):
    for template_function in template_functions:
        template.globals[template_function["name"]] = template_function["ptr"]
