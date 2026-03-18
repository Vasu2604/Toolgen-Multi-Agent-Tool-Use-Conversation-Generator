"""
generate_dataset.py
===================
Generates output/dataset.jsonl with 50 conversations using MockLLM.

MockLLM simulates a real LLM:
  - Clarification questions in ~50% of conversations
  - Endpoint-specific tool outputs (distinct per tool)
  - Endpoint-specific present_results messages
  - Final summary that references ALL tools used in the conversation
  - Varied user goals across domains
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolgen.registry  import ToolRegistry
from toolgen.graph     import ToolGraph, ToolChainSampler
from toolgen.memory    import MemoryStore
from toolgen.generator import ConversationGenerator, serialize_conversation


class MockLLM:
    """
    Deterministic mock LLM.

    Routing logic (in order):
      1. Opening message prompt        → cycling user opening
      2. should_clarify prompt         → question or NONE (alternates per conv)
      3. respond_to_clarification      → cycling clarification answer
      4. present_results prompt        → endpoint-specific summary
      5. generate_final_response       → synthesised summary referencing all tools
      6. respond_to_result / follow-up → cycling follow-up
      7. Fallback                      → generic reply

    generate_json routing:
      1. Planning prompt  → cycling user goal dict
      2. fill_arguments   → full args dict
      3. Tool execution   → endpoint-specific mock output
    """

    # ── Clarification questions keyed by required parameter name ──────
    _CLARIF_Q = {
        "city":           "Which city are you travelling to?",
        "country_code":   "What country are you in? (2-letter code, e.g. FR, US)",
        "origin":         "Which airport are you departing from?",
        "destination":    "Where are you flying to?",
        "departure_date": "What date are you planning to fly?",
        "check_in":       "When would you like to check in?",
        "from_currency":  "Which currency are you converting from?",
        "query":          "What exactly would you like me to search for?",
        "location":       "What location should I search near?",
        "date":           "What date do you need this for?",
        "title":          "What should I call this event?",
        "restaurant_id":  "Which restaurant are you asking about? I can search for options first.",
        "hotel_id":       "Which hotel are you asking about? I can search for options first.",
        "flight_id":      "Which flight are you asking about? I can search for flights first.",
        "amount":         "How much would you like to convert?",
        "party_size":     "How many people will be dining?",
    }

    _OPENINGS = [
        "Hi, I need some help planning my upcoming trip.",
        "Hey, can you help me with some travel research?",
        "I'm trying to plan something and could use your help.",
        "Could you help me with a few things I need to sort out?",
        "I need assistance with planning and booking.",
        "Can you look up some information for me?",
        "I'm organising a trip and need some help.",
        "I have a few tasks I'd like help with today.",
        "Could you research something for me?",
        "I'd like to plan a trip and need a hand.",
    ]

    _CLARIF_ANSWERS = [
        "It's Paris, France.",
        "I'm based in the US, so USD.",
        "The date is June 1st, 2025.",
        "It's for 2 people.",
        "I'd prefer a mid-range budget.",
        "I'm departing from New York, JFK airport.",
        "The destination is Paris, CDG.",
        "I'd like to check in on June 1st.",
        "I'm converting $500 USD to EUR.",
        "Let's call it 'Paris Planning Meeting'.",
    ]

    _FOLLOWUPS = [
        "That's very helpful, thank you! Can you continue?",
        "Great, that's what I needed. Please proceed.",
        "Perfect. What's next?",
        "Thanks! Please go ahead.",
        "That looks good. Keep going.",
        "Excellent, I appreciate that. Please continue.",
        "Good to know. What else can you find?",
        "That works for me, please go on.",
    ]

    def __init__(self):
        self._conv_count   = 0
        self._opening_idx  = 0
        self._answer_idx   = 0
        self._followup_idx = 0

    # ── generate() ────────────────────────────────────────────────────

    def generate(self, prompt: str, system: str = "") -> str:
        p = prompt.lower()

        # ── 1. Opening message ──────────────────────────────────
        if "first message" in p or ("write a natural" in p and "first message" in p):
            self._conv_count += 1
            msg = self._OPENINGS[self._opening_idx % len(self._OPENINGS)]
            self._opening_idx += 1
            return msg

        # ── 2. should_clarify prompt ────────────────────────────
        # Prompt structure from agents.py:
        #   "Look at this conversation and decide if there's a clarification question"
        #   "Next tool to call: {endpoint.name}"
        #   "Required parameters: [...]"
        #   "If any required parameter is missing ... write ONE specific clarification question."
        #   "If all required info is already in the conversation, write 'NONE'."
        if "next tool to call" in p and "required parameters" in p:
            # Only clarify on FIRST step of even-numbered conversations
            # and only if we haven't already clarified in this conversation
            if (self._conv_count % 2) == 0:
                for param, question in self._CLARIF_Q.items():
                    # Match param name in the required_parameters list
                    if f"'{param}'" in p or f'"{param}"' in p:
                        return question
                return "Could you provide more details about what you're looking for?"
            return "NONE"

        # ── 3. respond_to_clarification ─────────────────────────
        # Prompt from agents.py UserProxyAgent.respond_to_clarification:
        #   "You are a user with goal: ..."
        #   "The assistant asked: '{question}'"
        #   "Write a brief, helpful answer."
        if "the assistant asked" in p:
            ans = self._CLARIF_ANSWERS[self._answer_idx % len(self._CLARIF_ANSWERS)]
            self._answer_idx += 1
            return ans

        # ── 4. present_results ──────────────────────────────────
        # Prompt from agents.py AssistantAgent.present_results:
        #   "Tool called: {endpoint.name}"
        #   "User goal: ..."
        #   "Tool result: {...}"
        #   "Write a natural, helpful response..."
        if "tool called:" in p:
            if "search_flights" in p:
                return ("I found 2 flights from JFK to Paris. "
                        "Air France AF007 departs at 10:30am non-stop (7h30m) for $650, "
                        "or Delta DL080 offers a 1-stop option at $480. "
                        "I recommend the Air France direct flight for comfort.")
            if "get_flight_details" in p:
                return ("Your flight is Air France AF007, departing JFK at 10:30am "
                        "and arriving CDG at 10:00pm — a non-stop 7h30m Boeing 777 flight. "
                        "Includes meal service and 23kg baggage allowance.")
            if "search_hotels" in p:
                return ("I found 2 hotels in Paris. "
                        "Grand Hotel Paris (5-star, $280/night, rated 4.7) is a luxury option. "
                        "Hôtel Lutetia (4-star, $180/night, rated 4.5) offers great value. "
                        "Both are available for your dates.")
            if "get_hotel_reviews" in p:
                return ("Grand Hotel Paris has a 4.5/5 average from 234 guests. "
                        "Reviewers love the location, staff, and breakfast. "
                        "A recent guest wrote: 'Excellent location, impeccable service.'")
            if "get_current_weather" in p or "get_weather_forecast" in p:
                return ("Paris looks great on your travel date — partly cloudy, 22°C. "
                        "Humidity is 60% with a light 12 km/h breeze. "
                        "Perfect weather for sightseeing!")
            if "search_restaurants" in p:
                return ("I found excellent restaurants in Paris. "
                        "Le Jules Verne (4.8 stars, French cuisine) is near the Eiffel Tower. "
                        "L'Atelier Saint-Germain (4.6 stars) is another top choice.")
            if "check_reservation" in p:
                return ("Le Jules Verne has availability on June 1st for 2 guests. "
                        "Open slots are 19:00, 19:30, and 20:00. "
                        "I'd recommend the 19:30 slot for a beautiful evening view.")
            if "convert_currency" in p:
                return ("I've converted $500 USD to €460 EUR at today's rate of 0.92. "
                        "The EUR has been stable this week — a good time to convert.")
            if "get_exchange_rate" in p:
                return ("Current rates: 1 USD = 0.92 EUR, 0.79 GBP, 149.5 JPY. "
                        "The USD/EUR rate has been steady.")
            if "create_event" in p or "check_availability" in p:
                return ("Your event has been added to your calendar for June 1st, 10:00–11:00am. "
                        "You'll receive a reminder 30 minutes before.")
            if "search_news" in p or "get_top_headlines" in p:
                return ("Top headlines today: the EU AI Regulation Bill has passed, "
                        "OpenAI announced a new breakthrough, "
                        "and Paris has been named the top 2025 travel destination.")
            if "search_places" in p or "get_directions" in p:
                return ("I found 2 museums near central Paris. "
                        "The Louvre is 1.2km away, open until 9pm (rated 4.7). "
                        "Centre Pompidou is 1.8km away, open until 10pm (rated 4.5).")
            # Generic fallback for unrecognised endpoints
            return "I found the information you requested. Here are the key details."

        # ── 5. generate_final_response ──────────────────────────
        # Prompt from agents.py AssistantAgent.generate_final_response:
        #   "The user wanted: {user_goal}"
        #   "You made these tool calls and got results:"
        #   "- Called {endpoint} and got: ..."
        #   "Write a concise, helpful final summary..."
        #   "- Bring together all the information"
        if "you made these tool calls" in p or "bring together" in p:
            # Build a specific synthesis by detecting which tools were called
            parts = []
            if "flight_search" in p:
                parts.append("Air France AF007 departs JFK at 10:30am non-stop for $650")
            if "hotel_booking" in p:
                parts.append("Grand Hotel Paris is available at $280/night with a 4.7 rating")
            if "restaurant_finder" in p:
                parts.append("Le Jules Verne has a table available at 19:30 on June 1st")
            if "weather_api" in p:
                parts.append("Paris will be partly cloudy at 22°C — ideal weather")
            if "currency_exchange" in p:
                parts.append("$500 USD converts to €460 EUR at the current rate of 0.92")
            if "calendar_api" in p:
                parts.append("your event has been added to your calendar")
            if "news_api" in p:
                parts.append("the latest AI and travel headlines have been retrieved")
            if "maps_places" in p:
                parts.append("the Louvre and Centre Pompidou are nearby and open now")

            if parts:
                summary = "Here's a summary of everything I found: " + "; ".join(parts) + ". "
                summary += ("Let me know if you'd like to make any bookings "
                            "or need any further details!")
                return summary

            # Fallback if no specific tools detected
            return ("I've gathered all the information you needed for your trip. "
                    "Let me know if you'd like to proceed with any bookings "
                    "or have additional questions!")

        # ── 6. respond_to_result / follow-up ───────────────────
        # Prompt from agents.py UserProxyAgent.respond_to_result:
        #   "You are a user. The assistant said: '...'"
        #   "Write a brief natural follow-up."
        if "the assistant said" in p or "follow-up" in p or "follow up" in p:
            fu = self._FOLLOWUPS[self._followup_idx % len(self._FOLLOWUPS)]
            self._followup_idx += 1
            return fu

        # ── 7. Fallback ─────────────────────────────────────────
        return "Here are the results I found for your request."

    # ── generate_json() ───────────────────────────────────────────────

    def generate_json(self, prompt: str, system: str = "") -> dict:
        p = prompt.lower()

        # ── Planning prompt ─────────────────────────────────────
        if "user_goal" in p and "domain" in p:
            goals = [
                {"user_goal": "Plan a trip to Paris with flights and hotels",
                 "user_persona": "A 30-year-old professional planning a vacation",
                 "context": "Needs flights from NYC to Paris and a 5-star hotel.",
                 "ambiguities": ["exact dates", "budget"], "domain": "travel"},
                {"user_goal": "Find top restaurants and reserve a table in Tokyo",
                 "user_persona": "A food enthusiast visiting Japan for a week",
                 "context": "Looking for authentic Japanese dining for a party of 2.",
                 "ambiguities": ["cuisine type", "party size"], "domain": "food"},
                {"user_goal": "Monitor exchange rates and convert USD to EUR",
                 "user_persona": "An investor tracking global currency markets",
                 "context": "Needs to convert $500 and track rate movements.",
                 "ambiguities": ["exact amount", "target currencies"], "domain": "finance"},
                {"user_goal": "Get tech news headlines and schedule a team meeting",
                 "user_persona": "A product manager staying up-to-date with AI news",
                 "context": "Wants the latest AI news and a calendar slot for a team sync.",
                 "ambiguities": ["specific topics", "meeting time"], "domain": "news"},
                {"user_goal": "Find nearby museums and get directions from the hotel",
                 "user_persona": "A tourist spending 3 days in London",
                 "context": "Wants to visit cultural sites and navigate by public transport.",
                 "ambiguities": ["specific neighbourhood", "transport mode"], "domain": "maps"},
            ]
            import random
            rng = random.Random(hash(prompt[:100]) % 9999)
            return rng.choice(goals)

        # ── Argument-filling prompt ─────────────────────────────
        if "fill in the arguments" in p or "tool schema" in p:
            return {
                "city": "Paris", "country_code": "FR",
                "origin": "JFK", "destination": "CDG",
                "departure_date": "2025-06-01", "return_date": "2025-06-08",
                "check_in": "2025-06-01", "check_out": "2025-06-07",
                "query": "museums and art galleries",
                "location": "Paris, France",
                "from_currency": "USD", "to_currency": "EUR", "amount": 500,
                "date": "2025-06-01", "start_time": "10:00", "end_time": "11:00",
                "title": "Paris Team Meeting",
                "party_size": 2, "time": "19:30",
                "restaurant_id": "R001", "hotel_id": "H001",
                "flight_id": "F001", "result_id": "RES001",
                "limit": 5, "passengers": 1,
            }

        # ── Tool mock response ──────────────────────────────────
        # The execution prompt contains "Expected response fields:" and
        # "Endpoint: {name}\n" as the first line of that section.
        # We match on the exact header line to avoid false-positives from
        # session context (prior results) appended lower in the prompt.
        if "expected response fields" in p:

            def _ep(name: str) -> bool:
                """Match the current endpoint header line exactly."""
                return f"endpoint: {name}\n" in p

            if _ep("search_flights"):
                return {
                    "flights": [
                        {"flight_id": "F_001", "airline": "Air France",
                         "flight_number": "AF007", "origin": "JFK", "destination": "CDG",
                         "departure_time": "2025-06-01T10:30:00",
                         "arrival_time": "2025-06-01T22:00:00",
                         "duration": "7h30m", "stops": 0, "price": 650.0,
                         "currency": "USD", "available_seats": 12},
                        {"flight_id": "F_002", "airline": "Delta",
                         "flight_number": "DL080", "origin": "JFK", "destination": "CDG",
                         "departure_time": "2025-06-01T18:00:00",
                         "arrival_time": "2025-06-02T08:30:00",
                         "duration": "8h30m", "stops": 1, "price": 480.0,
                         "currency": "USD", "available_seats": 28},
                    ],
                    "total_results": 2, "status": "success",
                }

            if _ep("get_flight_details"):
                return {
                    "flight_id": "F_789", "airline": "Air France",
                    "flight_number": "AF007",
                    "departure_airport": "JFK", "arrival_airport": "CDG",
                    "departure_time": "2025-06-01T10:30:00",
                    "arrival_time": "2025-06-01T22:00:00",
                    "duration": "7h30m", "stops": 0, "aircraft": "Boeing 777",
                    "baggage_allowance": "23kg", "meal_service": True,
                    "price": 650.0, "status": "on_time",
                }

            if _ep("search_hotels"):
                return {
                    "hotels": [
                        {"hotel_id": "H_001", "name": "Grand Hotel Paris", "stars": 5,
                         "price_per_night": 280.0, "rating": 4.7, "reviews_count": 1234,
                         "address": "1 Rue de la Paix, Paris", "available": True},
                        {"hotel_id": "H_002", "name": "Hotel Lutetia", "stars": 4,
                         "price_per_night": 180.0, "rating": 4.5, "reviews_count": 876,
                         "address": "45 Boulevard Raspail, Paris", "available": True},
                    ],
                    "total_results": 2, "status": "success",
                }

            if _ep("get_hotel_reviews"):
                return {
                    "hotel_id": "H_456", "hotel_name": "Grand Hotel Paris",
                    "avg_rating": 4.5, "total_reviews": 234,
                    "reviews": [
                        {"reviewer": "John D.", "rating": 5,
                         "comment": "Excellent location, impeccable service.",
                         "date": "2025-05-01"},
                        {"reviewer": "Marie L.", "rating": 4,
                         "comment": "Beautiful hotel, a bit pricey but worth it.",
                         "date": "2025-04-15"},
                        {"reviewer": "Tom K.", "rating": 4,
                         "comment": "Great staff, comfortable rooms.",
                         "date": "2025-03-22"},
                    ],
                    "highlights": ["location", "staff", "breakfast"],
                    "status": "success",
                }

            if _ep("get_current_weather"):
                return {
                    "city": "Paris", "country": "FR",
                    "temperature": 22, "feels_like": 20,
                    "condition": "Partly Cloudy", "humidity": 60,
                    "wind_speed": 12, "wind_direction": "NW",
                    "visibility": 10, "uv_index": 5,
                    "forecast_date": "2025-06-01", "status": "success",
                }

            if _ep("get_weather_forecast"):
                return {
                    "city": "Paris",
                    "forecast": [
                        {"date": "2025-06-01", "high": 22, "low": 15,
                         "condition": "Partly Cloudy"},
                        {"date": "2025-06-02", "high": 24, "low": 16,
                         "condition": "Sunny"},
                        {"date": "2025-06-03", "high": 19, "low": 13,
                         "condition": "Rainy"},
                    ],
                    "status": "success",
                }

            if _ep("search_restaurants"):
                return {
                    "restaurants": [
                        {"restaurant_id": "R_001", "name": "Le Jules Verne",
                         "cuisine": "French", "rating": 4.8, "price_range": "$$$",
                         "open_now": True, "address": "Av. Gustave Eiffel, Paris"},
                        {"restaurant_id": "R_002", "name": "L'Atelier Saint-Germain",
                         "cuisine": "Modern French", "rating": 4.6, "price_range": "$$$",
                         "open_now": True, "address": "5 Rue de Montalembert, Paris"},
                    ],
                    "total_results": 2, "status": "success",
                }

            if _ep("check_reservation"):
                return {
                    "restaurant_id": "R_321", "restaurant_name": "Le Jules Verne",
                    "available": True,
                    "available_times": ["19:00", "19:30", "20:00"],
                    "party_size": 2, "date": "2025-06-01", "status": "success",
                }

            if _ep("convert_currency"):
                return {
                    "from_currency": "USD", "to_currency": "EUR",
                    "original_amount": 500, "converted_amount": 460.0,
                    "rate": 0.92, "timestamp": "2025-06-01T09:00:00",
                    "status": "success",
                }

            if _ep("get_exchange_rate"):
                return {
                    "base_currency": "USD",
                    "rates": {"EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "CAD": 1.36},
                    "timestamp": "2025-06-01T09:00:00", "status": "success",
                }

            if _ep("create_event"):
                return {
                    "event_id": "EVT_001", "title": "Paris Trip Planning",
                    "date": "2025-06-01", "start_time": "10:00", "end_time": "11:00",
                    "calendar": "Personal", "status": "created",
                    "confirmation": "Event successfully added to your calendar.",
                }

            if _ep("check_availability"):
                return {
                    "date": "2025-06-01", "start_time": "10:00", "end_time": "11:00",
                    "available": True, "conflicting_events": [],
                    "status": "success",
                }

            if _ep("get_top_headlines") or _ep("search_news"):
                return {
                    "headlines": [
                        {"title": "EU AI Regulation Bill Passes",
                         "source": "Reuters", "published": "2025-06-01"},
                        {"title": "OpenAI Announces New Research Breakthrough",
                         "source": "TechCrunch", "published": "2025-06-01"},
                        {"title": "Paris Named Top Travel Destination for 2025",
                         "source": "BBC Travel", "published": "2025-06-01"},
                    ],
                    "total_results": 3, "status": "success",
                }

            if _ep("search_places"):
                return {
                    "places": [
                        {"place_id": "P_001", "name": "Musee du Louvre",
                         "category": "Museum", "rating": 4.7,
                         "distance_km": 1.2, "open_now": True,
                         "address": "Rue de Rivoli, Paris", "hours": "09:00-21:00"},
                        {"place_id": "P_002", "name": "Centre Pompidou",
                         "category": "Museum", "rating": 4.5,
                         "distance_km": 1.8, "open_now": True,
                         "address": "Place Georges-Pompidou, Paris",
                         "hours": "11:00-22:00"},
                    ],
                    "total_results": 2, "status": "success",
                }

            if _ep("get_directions"):
                return {
                    "origin": "Grand Hotel Paris", "destination": "Musee du Louvre",
                    "distance_km": 1.2, "duration_minutes": 15,
                    "mode": "walking",
                    "steps": ["Head north on Rue de la Paix",
                              "Turn left onto Rue de Rivoli",
                              "The Louvre entrance is on your right"],
                    "status": "success",
                }

            # Unrecognised endpoint — minimal valid response
            return {"status": "success", "result": "mock_data", "id": "MOCK_001"}

        # Absolute fallback
        return {"status": "success", "result": "mock_data"}


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating output/dataset.jsonl")
    print("50 conversations · MockLLM · seed=42")
    print("Clarifications in ~50% of conversations")
    print("=" * 60)

    reg = ToolRegistry()
    reg.load_from_directory("data/sample_tools")

    graph   = ToolGraph(reg)
    sampler = ToolChainSampler(graph=graph, registry=reg, seed=42)
    memory  = MemoryStore(use_fallback=True)
    llm     = MockLLM()

    gen = ConversationGenerator(
        registry=reg, sampler=sampler, memory=memory, llm=llm,
        corpus_memory_enabled=True, seed=42,
    )
    gen.planner_agent.llm = llm
    gen.user_proxy.llm    = llm
    gen.assistant.llm     = llm
    gen.executor.llm      = llm

    convs = gen.generate_batch(50)

    os.makedirs("output", exist_ok=True)
    path = "output/dataset.jsonl"
    with open(path, "w") as f:
        for c in convs:
            f.write(json.dumps(serialize_conversation(c)) + "\n")

    total_tc    = sum(len(c.tool_calls) for c in convs)
    with_clarif = sum(1 for c in convs if c.num_clarification_questions > 0)
    mgr_vals    = [c.memory_grounding_rate for c in convs
                   if c.memory_grounding_rate is not None]
    patterns    = {}
    for c in convs:
        patterns[c.pattern_type] = patterns.get(c.pattern_type, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"✅ Wrote {len(convs)} conversations → {path}")
    print(f"   avg tool calls / conv  : {total_tc / len(convs):.1f}")
    print(f"   pct_with_clarification : {with_clarif / len(convs) * 100:.0f}%")
    print(f"   avg memory_grounding   : {sum(mgr_vals)/len(mgr_vals):.2f}")
    print(f"   patterns               : {patterns}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()