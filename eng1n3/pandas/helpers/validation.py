"""
Definition of the validation functions for the Pandas engine
(c) 2023 tsm
"""
import pandas as pd
from typing import List, Union, Tuple

from f3atur3s import TensorDefinition, FeatureExpander, Feature, FeatureGrouper, FeatureHelper, FeatureTypeTimeBased
from f3atur3s import LEARNING_CATEGORY_NONE

from ..common.exception import EnginePandasException


class EnginePandasValidation:
    @staticmethod
    def val_features_in_data_frame(df: pd.DataFrame, tensor_def: TensorDefinition, one_hot_prefix: str):
        """
        Validation function to check if all features in a tensor definition are known data frame columns

        Args:
            df: A Panda Dataframe
            tensor_def: A tensor definition
            one_hot_prefix: The one_hot_prefix used in the engine set-up

        Returns:
             None
        """
        for feature in tensor_def.features:
            if isinstance(feature, FeatureExpander):
                names = [name for name in df.columns
                         if name.startswith(feature.base_feature.name + one_hot_prefix)]
            else:
                names = [name for name in df.columns if name == feature.name]
            if len(names) == 0:
                raise EnginePandasException(
                    f'During reshape, all features of tensor definition must be in the panda. Missing {feature.name}'
                )

    @staticmethod
    def val_single_date_format_code(format_codes: List[str]):
        """
        Validation function to check that there is a single format code for dates across a set of format codes.

        Args:
            format_codes: List of format codes from feature definitions. Is a string

        Raises:
            EnginePandasException: if the validation condition is not met

        Returns:
             None
        """
        if len(format_codes) > 1:
            raise EnginePandasException(f'All date formats should be the same. Got <{format_codes}> different codes')

    @staticmethod
    def val_ready_for_inference(tensor_def: TensorDefinition, inference: bool):
        """
        Validation function to check if all feature are ready for inference. Some features have specific inference
        attributes that need to be set before an inference file can be made.

        Args:
            tensor_def: The tensor that needs to be ready for inference.
            inference: Indication if we are inference mode or not.

        Raises:
            EnginePandasException: if the validation condition is not met

        Returns:
             None
        """
        if inference:
            if not tensor_def.inference_ready:
                raise EnginePandasException(
                    f'Tensor <{tensor_def.name}> not ready for inference. Following features not ready ' +
                    f' {tensor_def.features_not_inference_ready()}'
                )

    @staticmethod
    def val_features_defined_as_columns(df: pd.DataFrame, features: List[Feature]):
        """
        Validation function that checks if the needed columns are available in the Panda. Only root features which
        are not derived from other features need to be in the Panda. The rest of the features can obviously be
        derived from the root features.

        Args:
            df: The base Panda data to be checked.
            features: List of feature to check.

        Raises:
            EnginePandasException: if the validation condition is not met

        Returns:
             None
        """
        root_features = [f for f in features if len(f.embedded_features) == 0]
        unknown_features = [f for f in root_features if f.name not in df.columns]
        if len(unknown_features) != 0:
            raise EnginePandasException(
                f'All root features of a tensor definition (i.e. non-derived features) must be in the input df. Did '
                f'not find {[f.name for f in unknown_features]}'
            )

    @staticmethod
    def val_time_feature_needed_non_series(target_tensor_def: TensorDefinition, time_feature: Feature):
        """
        Validation function that checks if a "time_feature" variable will be required to build the features. This can
        be the case for instance for Feature Groupers, they need to know about time.

        Args:
            target_tensor_def: The TensorDefinition we are going to try and build.
            time_feature: The time_feature variable that was provided during the engine call. (Can be None)

        Raises:
            EnginePandasException: if the validation condition is not met

        Returns:
            None
        """
        if len(FeatureHelper.filter_feature(FeatureGrouper, target_tensor_def.embedded_features)) > 0:
            if time_feature is None:
                raise EnginePandasException(
                    f'There is a FeatureGrouper in the Tensor Definition to create. They need a time field to ' +
                    f' process. Please provide the parameter ''time_feature''.'
                )
            else:
                if not isinstance(time_feature.type, FeatureTypeTimeBased):
                    raise EnginePandasException(
                        f'The time feature used to build a series must be date based. It is of type {time_feature.type}'
                    )

    @staticmethod
    def val_time_feature_needed_series(time_feature: Feature):
        """
        Validation function that checks if a "time_feature" variable is available to build a Series.

        Args:
            time_feature: The time_feature variable that was provided during the engine call. (Can be None)

        Raises:
            EnginePandasException: if the validation condition is not met

        Returns:
            None
        """
        if time_feature is None:
            raise EnginePandasException(
                f'There is a FeatureGrouper in the Tensor Definition to create. They need a time field to ' +
                f' process. Please provide the parameter ''time_feature''.'
            )
        else:
            if not isinstance(time_feature.type, FeatureTypeTimeBased):
                raise EnginePandasException(
                    f'The time feature used to build a series must be date based. It is of type {time_feature.type}'
                )

    @staticmethod
    def val_all_same_learning_category(
            target_tensor_def: Union[TensorDefinition, Tuple[TensorDefinition, ...]]) -> None:
        """
        Validation routine to check that all the features of the TensorDefinition have the same learning
        category

        Args:
            target_tensor_def: The TensorDefinition(s) we are going to try and build.

        Returns:
             None
        """
        for td in (target_tensor_def,) if isinstance(target_tensor_def, TensorDefinition) else target_tensor_def:
            lcs = list(set([f.learning_category for f in td.features]))
            if len(lcs) > 1:
                raise EnginePandasException(f'All embedded features in the TensorDefinition {td.name} ' +
                                            f'should have the same LearningCategories. Found LCs {lcs}')

    @staticmethod
    def val_no_none_learning_category(
            target_tensor_def: Union[TensorDefinition, Tuple[TensorDefinition, ...]]) -> None:
        """
        Validation routine to check that none of the features of the TensorDefinition have LEARNING_CATEGORY_NONE

        Args:
            target_tensor_def: The TensorDefinition(s) we are going to try and build.

        Raises:
            EnginePandasException if there is a feature with LEARNING_CATEGORY_NONE

        Returns:
             None
        """
        for td in (target_tensor_def,) if isinstance(target_tensor_def, TensorDefinition) else target_tensor_def:
            if LEARNING_CATEGORY_NONE in [f.learning_category for f in td.features]:
                raise EnginePandasException(
                    f'Can not build features with Learning Category None ' +
                    f'{[f.name for f in td.features if f.learning_category == LEARNING_CATEGORY_NONE]}'
                )

    @staticmethod
    def val_same_feature_root_type(target_tensor_def: Union[TensorDefinition, Tuple[TensorDefinition, ...]]) -> None:
        """
        Validation routine to check that all the features of the TensorDefinition have the same FeatureRootType

        Args:
            target_tensor_def: The TensorDefinition we are going to try and build.

        Raises:
            EnginePandasException if the feature do not all have the same FeatureRootType

        Returns:
             None
        """
        for td in (target_tensor_def,) if isinstance(target_tensor_def, TensorDefinition) else target_tensor_def:
            rt = list(set([f.type.root_type for f in td.features]))
            if len(rt) > 1:
                raise EnginePandasException(
                    f'Found more than one feature root type. {rt} in TensorDefinition {td.name}. ' +
                    f'This process can only handle feature of the same root type, for instance only int or only float'
                )

    @staticmethod
    def val_one_feature_per_td(target_tensor_def: Union[TensorDefinition, Tuple[TensorDefinition, ...]]):
        """
        Validation routine to check that there is only one Feature per each input Target TensorDefinition

        Args:
            target_tensor_def: The TensorDefinition we are going to try and build.

        Raises:
            EnginePandasException if there is more than one feature in a TensorDefinition

        Returns:
             None
        """
        for td in (target_tensor_def,) if isinstance(target_tensor_def, TensorDefinition) else target_tensor_def:
            if len(td.features) > 1:
                raise EnginePandasException(
                    f'The TensorDefinition with name {td.name} has more than one feature. Please provide ' +
                    f'TensorDefinitions with only one feature in them'
                )

    @staticmethod
    def val_tds_contain_same_feature_class(target_tensor_def: Union[TensorDefinition, Tuple[TensorDefinition, ...]]):
        """
        Validation routine to check that all provided TensorDefinitions only contain one single feature class

        Args:
            target_tensor_def: The TensorDefinition we are going to try and build.

        Raises:
            EnginePandasException if there is more than one unique feature class across the TensorDefinitions

        Returns:
             None
        """
        ttd = (target_tensor_def,) if isinstance(target_tensor_def, TensorDefinition) else target_tensor_def
        types = set([type(f) for td in ttd for f in td.features])
        if len(types) > 1:
            raise EnginePandasException(
                f'Found more than one unique feature class in the TensorDefinitions. Got {[t.__name__ for t in types]}'
                + f'. Please provide only one feature class across all the TensorDefinitions'
            )
