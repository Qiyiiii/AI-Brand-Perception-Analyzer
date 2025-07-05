from agent import Evaluation_agent
from utils import ModelEndpoint
import sqlite3
from info_getter import get_records_by_models_all,get_most_asked_aspect, plot_company_trend_over_time



if __name__ == "__main__":
    p = plot_company_trend_over_time("eval_cache.db")
    print(p)