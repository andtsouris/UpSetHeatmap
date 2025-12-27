import typing

import numpy as np
import pandas as pd


def _aggregate_data(df: pd.DataFrame, subset_size: str, sum_over: str | bool | None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns
    -------
    df : DataFrame
        full data frame
    aggregated : Series
        aggregates
    aggregated_byGroup: pd.DataFrame
        Aggregates per group
    group_size : Series
    """
    _SUBSET_SIZE_VALUES = ["auto", "count", "sum"]
    if subset_size not in _SUBSET_SIZE_VALUES:
        raise ValueError(
            f"subset_size should be one of {_SUBSET_SIZE_VALUES}."
            f" Got {repr(subset_size)}"
        )
    
    if sum_over is False:
        raise ValueError("Unsupported value for sum_over: False")
    elif subset_size == "auto" and sum_over is None:
        sum_over = False
    elif subset_size == "count":
        if sum_over is not None:
            raise ValueError(
                "sum_over cannot be set if subset_size=%r" % subset_size
            )
        sum_over = False
    elif subset_size == "sum" and sum_over is None:
        raise ValueError(
            "sum_over should be a field name if "
            'subset_size="sum" and a DataFrame is '
            "provided."
        )

    # Group_by for the aggregation
    gb = df.groupby(level=list(range(df.index.nlevels)), sort=False)
    # Group_by for the aggregation in each individual group
    df_byGroup = df.copy()
    df_byGroup.index = pd.MultiIndex.from_frame(df_byGroup.index.to_frame().reset_index(drop=True).assign(group=df_byGroup['group'].values))
    df_byGroup = df_byGroup.drop(['group', 'index'], axis=1)
    gb_byGroup = df_byGroup.groupby(level=list(range(df_byGroup.index.nlevels)), sort=False)
    if sum_over is False:
        aggregated = gb.size()
        aggregated.name = "size"
        aggregated_byGroup = gb_byGroup.size()
        group_sizes = df.groupby('group').size()

    elif hasattr(sum_over, "lower"):
        aggregated = gb[sum_over].sum()
        aggregated_byGroup = gb_byGroup[sum_over].sum()
        group_sizes = df.groupby('group')['value'].sum()
    else:
        raise ValueError("Unsupported value for sum_over: %r" % sum_over)

    return df, aggregated, aggregated_byGroup, group_sizes


def _check_index(df: pd.DataFrame) -> pd.DataFrame:
    # check all indices are boolean
    if not all({True, False} >= set(level) for level in df.index.levels):
        raise ValueError(
            "The DataFrame has values in its index that are not " "boolean"
        )
    df = df.copy(deep=False)
    kw = {
        "levels": [x.astype(bool) for x in df.index.levels],
        "names": df.index.names,
    }
    if hasattr(df.index, "codes"):
        # compat for pandas <= 0.20
        kw["codes"] = df.index.codes
    else:
        kw["labels"] = df.index.labels
    df.index = pd.MultiIndex(**kw)
    return df


def _scalar_to_list(val):
    if not isinstance(val, (typing.Sequence, set)) or isinstance(val, str):
        val = [val]
    return val


def _check_percent(value: str | int | float, agg: pd.Series) -> float | int | None:
    if not isinstance(value, str):
        return value
    try:
        if value.endswith("%") and 0 <= float(value[:-1]) <= 100:
            return float(value[:-1]) / 100 * agg.sum()
    except ValueError:
        pass
    raise ValueError(
        f"String value must be formatted as percentage between 0 and 100. Got {value}"
    )


class QueryResult:
    """Container for reformatted data and aggregates

    Attributes
    ----------
    data : DataFrame
        Selected samples. The index is a MultiIndex with one boolean level for
        each category.
    subsets : dict[frozenset, DataFrame]
        Dataframes for each intersection of categories.
    subset_sizes : Series
        Total size of each selected subset as a series. The index is as
        for `data`.
    category_totals : Series
        Total size of each category, regardless of selection.
    total : number
        Total number of samples, or sum of sum_over value.
    """

    def __init__(self, data: pd.DataFrame, subset_sizes: pd.Series,
                 category_totals: pd.Series, group_totals: pd.Series, group_agg: pd.Series, total: int | float):
        self.data = data
        self.subset_sizes = subset_sizes
        self.category_totals = category_totals
        self.group_totals = group_totals
        self.group_agg = group_agg
        self.total = total

    def __repr__(self):
        return (
            "QueryResult(data={data}, "
            "subset_sizes={subset_sizes}, group_totals={group_totals}, "
            "category_totals={category_totals}, total={total}".format(**vars(self))
        )

    @property
    def subsets(self):
        categories = np.asarray(self.data.index.names)
        return {    
            frozenset(categories.take(mask)): subset_data
            for mask, subset_data in self.data.groupby(
                level=list(range(len(categories))), sort=False
            )
        }


def query(
    data,
    present=None,
    absent=None,
    min_subset_size=None,
    max_subset_size=None,
    max_subset_rank=None,
    min_degree=None,
    max_degree=None,
    sort_by="degree",
    sort_categories_by="cardinality",
    sort_groups_by=None,
    group_order=None,
    subset_size="auto",
    sum_over=None,
    include_empty_subsets=False,
):
# TODO: Add parameters for group sorting to the dostring
    """Transform and filter a categorised dataset

    Retrieve the set of items and totals corresponding to subsets of interest.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Elements associated with categories (a DataFrame), or the size of each
        subset of categories (a Series).
        Should have MultiIndex where each level is binary,
        corresponding to category membership.
        If a DataFrame, `sum_over` must be a string or False.
    present : str or list of str, optional
        Category or categories that must be present in subsets for styling.
    absent : str or list of str, optional
        Category or categories that must not be present in subsets for
        styling.
    min_subset_size : int or "number%", optional
        Minimum size of a subset to be reported. All subsets with
        a size smaller than this threshold will be omitted from
        category_totals and data.  This may be specified as a percentage
        using a string, like "50%".
        Size may be a sum of values, see `subset_size`.

        .. versionchanged:: 0.9
            Support percentages
    max_subset_size : int or "number%", optional
        Maximum size of a subset to be reported.

        .. versionchanged:: 0.9
            Support percentages
    max_subset_rank : int, optional
        Limit to the top N ranked subsets in descending order of size.
        All tied subsets are included.

        .. versionadded:: 0.9
    min_degree : int, optional
        Minimum degree of a subset to be reported.
    max_degree : int, optional
        Maximum degree of a subset to be reported.
    sort_by : {'cardinality', 'degree', '-cardinality', '-degree',
               'input', '-input'}
        If 'cardinality', subset are listed from largest to smallest.
        If 'degree', they are listed in order of the number of categories
        intersected. If 'input', the order they appear in the data input is
        used.
        Prefix with '-' to reverse the ordering.

        Note this affects ``subset_sizes`` but not ``data``.
    sort_categories_by : {'cardinality', '-cardinality', 'input', '-input'}
        Whether to sort the categories by total cardinality, or leave them
        in the input data's provided order (order of index levels).
        Prefix with '-' to reverse the ordering.
    subset_size : {'auto', 'count', 'sum'}
        Configures how to calculate the size of a subset. Choices are:

        'auto' (default)
            If `data` is a DataFrame, count the number of rows in each group,
            unless `sum_over` is specified.
            If `data` is a Series with at most one row for each group, use
            the value of the Series. If `data` is a Series with more than one
            row per group, raise a ValueError.
        'count'
            Count the number of rows in each group.
        'sum'
            Sum the value of the `data` Series, or the DataFrame field
            specified by `sum_over`.
    sum_over : str or None
        If `subset_size='sum'` or `'auto'`, then the intersection size is the
        sum of the specified field in the `data` DataFrame. If a Series, only
        None is supported and its value is summed.
    include_empty_subsets : bool (default=False)
        If True, all possible category combinations will be returned in
        subset_sizes, even when some are not present in data.

    Returns
    -------
    QueryResult
        Including filtered ``data``, filtered and sorted ``subset_sizes`` and
        overall ``category_totals`` and ``total``.

    Examples
    --------
    >>> from upsetheatmap import query, generate_samples
    >>> data = generate_samples(n_samples=20)
    >>> result = query(data, present="cat1", max_subset_size=4)
    >>> result.category_totals
    cat1    14
    cat2     4
    cat0     0
    dtype: int64
    >>> result.subset_sizes
    cat1  cat2  cat0
    True  True  False    3
    Name: size, dtype: int64
    >>> result.data
                     index     value
    cat1 cat2 cat0
    True True False      0  2.04...
              False      2  2.05...
              False     10  2.55...
    >>>
    >>> # Sorting:
    >>> query(data, min_degree=1, sort_by="degree").subset_sizes
    cat1   cat2   cat0
    True   False  False    11
    False  True   False     1
    True   True   False     3
    Name: size, dtype: int64
    >>> query(data, min_degree=1, sort_by="cardinality").subset_sizes
    cat1   cat2   cat0
    True   False  False    11
           True   False     3
    False  True   False     1
    Name: size, dtype: int64
    >>>
    >>> # Getting each subset's data
    >>> result = query(data)
    >>> result.subsets[frozenset({"cat1", "cat2"})]
                index     value
    cat1  cat2 cat0
    False True False      3  1.333795
    >>> result.subsets[frozenset({"cat1"})]
                        index     value
    cat1  cat2  cat0
    False False False      5  0.918174
                False      8  1.948521
                False      9  1.086599
                False     13  1.105696
                False     19  1.339895
    """

    data, agg, group_agg, group_sizes = _aggregate_data(data, subset_size, sum_over)
    data = _check_index(data)
    grand_total = agg.sum()
    category_totals = [
        agg[agg.index.get_level_values(name).values.astype(bool)].sum()
        for name in agg.index.names
    ]
    category_totals = pd.Series(category_totals, index=agg.index.names)
    group_totals = data.groupby('group')['value'].sum()
    
    if include_empty_subsets:
        nlevels = len(agg.index.levels)
        if nlevels > 10:
            raise ValueError(
                "include_empty_subsets is supported for at most 10 categories"
            )
        new_agg = pd.Series(
            0,
            index=pd.MultiIndex.from_product(
                [[False, True]] * nlevels, names=agg.index.names
            ),
            dtype=agg.dtype,
            name=agg.name,
        )
        new_agg.update(agg)
        agg = new_agg
        
    # Add sorting of the groups
    if sort_groups_by == "count":
        group_order = group_sizes.sort_values(ascending=False).index.tolist()
    elif sort_groups_by == "custom":
        if not set(group_order) == set(group_sizes.index.tolist()):
            raise ValueError(f"group_order must be list containing all the group names: {group_sizes.index.tolist()}")
    elif group_order == None:
        group_order = group_sizes.index.tolist()
    else:
        raise ValueError("sort_groups_by must be one of {'count', 'custom', None}")

    # LATER: Pass the group size and rank parameters to _filter_subsets
    # LATER: Add sorting of the groups
    if sort_categories_by in ("cardinality", "-cardinality"):
        category_totals.sort_values(
            ascending=sort_categories_by[:1] == "-", inplace=True
        )
    elif sort_categories_by == "-input":
        category_totals = category_totals[::-1]
    elif sort_categories_by in (None, "input"):
        pass
    else:
        raise ValueError("Unknown sort_categories_by: %r" % sort_categories_by)
    data = data.reorder_levels(category_totals.index.values)
    agg = agg.reorder_levels(category_totals.index.values)

    if sort_by in ("cardinality", "-cardinality"):
        agg = agg.sort_values(ascending=sort_by[:1] == "-")
    elif sort_by in ("degree", "-degree"):
        index_tuples = sorted(
            agg.index,
            key=lambda x: (sum(x),) + tuple(reversed(x)),
            reverse=sort_by[:1] == "-",
        )
        agg = agg.reindex(
            pd.MultiIndex.from_tuples(index_tuples, names=agg.index.names)
        )
    elif sort_by == "-input":
        agg = agg[::-1]
    elif sort_by in (None, "input"):
        pass
    else:
        raise ValueError("Unknown sort_by: %r" % sort_by)

    # Implement group_agg reorder to fit the index orders of agg and group_totals
    combined_index = []
    for g in group_order:
        for idx in agg.index:
            combined_index.append(idx+(g,))
    combined_index

    group_agg = group_agg.reindex(pd.MultiIndex.from_tuples(combined_index, names=group_agg.index.names))
    group_agg = group_agg.fillna(0)

    return QueryResult(
        data=data,
        subset_sizes=agg,
        group_agg=group_agg,
        category_totals=category_totals,
        group_totals=group_totals,
        total=grand_total
    )
