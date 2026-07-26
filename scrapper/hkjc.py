"""HKJC GraphQL provider adapter.

The endpoint may reject requests based on network location. Provider errors are
raised intact so callers never mistake stale or missing data for a valid race.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "https://info.cld.hkjc.com/graphql/base/"

RACE_SNAPSHOT_QUERY = r"""
query RaceSnapshot(
  $date: String,
  $venueCode: String,
  $raceNo: Int,
  $oddsTypes: [OddsType]
) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    id
    venueCode
    date
    status
    races {
      id
      no
      status
      postTime
      distance
      wageringFieldSize
      claCode
      raceClass_en
      go_en
      raceCourse { displayCode description_en }
      runners {
        id
        no
        standbyNo
        status
        name_en
        barrierDrawNumber
        handicapWeight
        currentWeight
        currentRating
        internationalRating
        gearInfo
        allowance
        last6run
        saddleClothNo
        winOdds
        horse { id code }
        jockey { code name_en }
        trainer { code name_en }
      }
    }
    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
      id
      status
      sellStatus
      oddsType
      lastUpdateTime
      minTicketCost
      oddsNodes {
        combString
        oddsValue
        hotFavourite
        oddsDropValue
      }
    }
  }
}
"""


class HKJCError(RuntimeError):
    """Base error for provider failures."""


class HKJCAccessError(HKJCError):
    """The endpoint rejected or could not receive the request."""


class HKJCResponseError(HKJCError):
    """The endpoint returned GraphQL errors or an invalid payload."""


@dataclass
class HKJCClient:
    endpoint: str = DEFAULT_ENDPOINT
    timeout_seconds: float = 20.0

    def request(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "iMa-live-race-collector/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HKJCAccessError(f"HKJC HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise HKJCAccessError(f"HKJC request failed: {exc.reason}") from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HKJCResponseError("HKJC returned non-JSON content") from exc

        if result.get("errors"):
            messages = "; ".join(str(item.get("message", item)) for item in result["errors"])
            raise HKJCResponseError(f"HKJC GraphQL error: {messages}")
        if not isinstance(result.get("data"), dict):
            raise HKJCResponseError("HKJC response did not contain a data object")
        return result

    def fetch_race_snapshot(
        self,
        race_no: int,
        race_date: str | None = None,
        venue_code: str | None = None,
        odds_types: tuple[str, ...] = ("WIN", "PLA"),
    ) -> dict[str, Any]:
        return self.request(
            RACE_SNAPSHOT_QUERY,
            {
                "date": race_date,
                "venueCode": venue_code,
                "raceNo": race_no,
                "oddsTypes": list(odds_types),
            },
        )
