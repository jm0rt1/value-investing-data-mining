
# -*- coding: utf-8 -*-

from src.value_investing_strategy.strategy_system.analysis.Analyzer import Analyzer
import src.value_investing_strategy.strategy_system.settings.Settings


class QFAnalyzer(Analyzer):

    def __init__(self, Settings: src.value_investing_strategy.strategy_system.settings.Settings.Settings):
        self.uid = None

    def to_csv(self, ):
        pass

    def find_stocks(self, ):
        pass
