import pandas as pd
from steerability_eval.eval.base import BaseEval

class Scorer:
    def __init__(self, eval: BaseEval):
        self.eval = eval
        self.scores_df = pd.DataFrame(self.eval.scores)
        
        # Sort both index and columns in the same order
        all_personas = sorted(self.scores_df.index)
        self.scores_df = self.scores_df.reindex(index=all_personas, columns=all_personas)
        
        self.row_percentiles, self.col_percentiles = self.get_percentiles()
        self.sensitivity, self.sensitivity_scores = self.get_sensitivity_scores()
        self.specificity, self.specificity_scores = self.get_specificity_scores()
        self.results_df = pd.DataFrame({
            'sensitivity': self.sensitivity_scores,
            'specificity': self.specificity_scores
        })

    def get_percentiles(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        row_percentiles: pd.DataFrame = self.scores_df.apply(lambda row: row.rank(pct=True, method='max'), axis=1) # type: ignore
        col_percentiles: pd.DataFrame = self.scores_df.apply(lambda col: col.rank(pct=True, method='max'), axis=0)
        return row_percentiles, col_percentiles

    def get_sensitivity_scores(self, aggregation: str = 'mean') -> tuple[float, pd.Series]:
        sensitivities = pd.Series({
            persona_id: self.col_percentiles[persona_id].loc[persona_id]
            for persona_id in self.col_percentiles.columns
        })
        if aggregation == 'mean':
            sensitivity = sensitivities.mean()
        elif aggregation == 'median':
            sensitivity = sensitivities.median()
        else:
            raise ValueError(f'Invalid aggregation method: {aggregation}')
        return sensitivity, sensitivities

    def get_specificity_scores(self, aggregation: str = 'mean') -> tuple[float, pd.Series]:
        specificities = pd.Series({
            persona_id: self.row_percentiles[persona_id].loc[persona_id]
            for persona_id in self.row_percentiles.index
        })
        if aggregation == 'mean':
            specificity = specificities.mean()
        elif aggregation == 'median':
            specificity = specificities.median()
        else:
            raise ValueError(f'Invalid aggregation method: {aggregation}')
        return specificity, specificities
