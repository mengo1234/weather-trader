from datetime import datetime

from pydantic import BaseModel


class MarketQuery(BaseModel):
    question: str
    outcomes: list[str]
    outcome_prices: list[float]
    market_id: str | None = None
    expiry: datetime | None = None


class OutcomePrediction(BaseModel):
    outcome: str
    market_price: float
    our_probability: float
    edge: float
    confidence: float


class BettingRecommendation(BaseModel):
    market_question: str
    city: str
    variable: str
    date: str
    outcomes: list[OutcomePrediction]
    best_bet: str | None = None
    kelly_fraction: float = 0.0
    suggested_size_pct: float = 0.0
    expected_value: float = 0.0
    reasoning: str = ""


class BetRecordRequest(BaseModel):
    market_question: str
    outcome: str
    stake: float = 0
    odds: float = 0
    our_prob: float = 0
    edge: float = 0
    confidence: float = 0
    city_slug: str = ""
    target_date: str = ""
    market_prediction_id: int | None = None
    variable: str = ""
    confidence_scores_json: str | None = None
    cross_ref_json: str | None = None


class BetResolveRequest(BaseModel):
    won: bool


class BetImportItem(BaseModel):
    timestamp: str = ""
    market_question: str = ""
    outcome: str = ""
    stake: float = 0
    odds: float = 0
    our_prob: float = 0
    edge: float = 0
    confidence: float = 0
    city_slug: str = ""
    target_date: str = ""
    status: str = "pending"
    pnl: float = 0


class BetImportRequest(BaseModel):
    bets: list[BetImportItem]


class MarketPrediction(BaseModel):
    market_id: str
    question: str
    predicted_at: datetime
    city_slug: str
    variable: str
    target_date: str
    outcomes: list[OutcomePrediction]
    recommendation: BettingRecommendation | None = None
