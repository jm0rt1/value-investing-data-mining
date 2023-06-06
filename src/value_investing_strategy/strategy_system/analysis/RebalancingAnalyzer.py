
# -*- coding: utf-8 -*-

from src.value_investing_strategy.strategy_system.analysis.Analyzer import Analyzer
from src.value_investing_strategy.strategy_system.settings.RebalancingAnalyzerSettings import RebalancingAnalyzerSettings


class RebalancingAnalyzer(Analyzer):

    def __init__(self, settings: RebalancingAnalyzerSettings):
        self.uid = None

    def save(self, ):
        pass
