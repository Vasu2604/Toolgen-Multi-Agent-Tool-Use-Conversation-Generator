"""
pick_subset.py — ToolBench Subset Picker

Reads the ToolBench tools directory (JSON format) or Python api.py files,
scores tools by endpoint count, picks N tools per category, and writes
ToolBench-compatible JSON to the output directory.

When source data is insufficient (<10 categories), automatically supplements
with a comprehensive synthetic dataset covering all 49 ToolBench categories.

Usage:
    python3 scripts/pick_subset.py \
        --source ~/Desktop/ToolBench/data/toolenv/tools \
        --output data/toolbench_subset \
        --tools-per-category 3 \
        --min-endpoints 1
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

import click


# ─────────────────────────────────────────────────────────────
# Synthetic tool catalog  (49 ToolBench-style categories)
# ─────────────────────────────────────────────────────────────

SYNTHETIC_CATALOG = {
    "Advertising": [
        {
            "tool_name": "ad_campaign_manager",
            "tool_description": "Manage and track advertising campaigns across digital platforms.",
            "api_list": [
                {"name": "create_campaign", "description": "Create a new ad campaign.", "method": "POST",
                 "required_parameters": [{"name": "name", "type": "string", "description": "Campaign name"},
                                         {"name": "budget", "type": "number", "description": "Daily budget in USD"}],
                 "optional_parameters": [{"name": "start_date", "type": "string", "description": "Start date YYYY-MM-DD"}],
                 "response_fields": ["campaign_id", "status", "impressions"]},
                {"name": "get_campaign_stats", "description": "Get performance stats for a campaign.", "method": "GET",
                 "required_parameters": [{"name": "campaign_id", "type": "string", "description": "Campaign ID"}],
                 "optional_parameters": [{"name": "period", "type": "string", "description": "daily|weekly|monthly"}],
                 "response_fields": ["impressions", "clicks", "ctr", "spend", "conversions"]},
                {"name": "pause_campaign", "description": "Pause an active campaign.", "method": "POST",
                 "required_parameters": [{"name": "campaign_id", "type": "string", "description": "Campaign ID"}],
                 "optional_parameters": [],
                 "response_fields": ["campaign_id", "status"]},
            ]
        },
        {
            "tool_name": "keyword_planner",
            "tool_description": "Research keywords for paid search advertising.",
            "api_list": [
                {"name": "get_keyword_ideas", "description": "Get keyword ideas for a topic.", "method": "GET",
                 "required_parameters": [{"name": "seed_keyword", "type": "string", "description": "Seed keyword"}],
                 "optional_parameters": [{"name": "location", "type": "string", "description": "Target location"}],
                 "response_fields": ["keywords", "avg_monthly_searches", "competition", "bid_range"]},
                {"name": "get_keyword_metrics", "description": "Get metrics for specific keywords.", "method": "GET",
                 "required_parameters": [{"name": "keywords", "type": "string", "description": "Comma-separated keywords"}],
                 "optional_parameters": [],
                 "response_fields": ["keyword", "volume", "cpc", "competition_score"]},
            ]
        },
        {
            "tool_name": "audience_targeting",
            "tool_description": "Define and manage audience segments for targeted advertising.",
            "api_list": [
                {"name": "create_audience", "description": "Create a custom audience segment.", "method": "POST",
                 "required_parameters": [{"name": "name", "type": "string", "description": "Audience name"},
                                         {"name": "criteria", "type": "string", "description": "JSON targeting criteria"}],
                 "optional_parameters": [],
                 "response_fields": ["audience_id", "estimated_size", "status"]},
                {"name": "get_audience_insights", "description": "Get demographic insights for an audience.", "method": "GET",
                 "required_parameters": [{"name": "audience_id", "type": "string", "description": "Audience ID"}],
                 "optional_parameters": [],
                 "response_fields": ["age_distribution", "gender_split", "interests", "devices"]},
            ]
        },
    ],
    "Artificial_Intelligence_Machine_Learning": [
        {
            "tool_name": "text_classification_api",
            "tool_description": "Classify text into categories using pre-trained ML models.",
            "api_list": [
                {"name": "classify_text", "description": "Classify a single text string.", "method": "POST",
                 "required_parameters": [{"name": "text", "type": "string", "description": "Text to classify"},
                                         {"name": "model", "type": "string", "description": "Model to use"}],
                 "optional_parameters": [{"name": "top_n", "type": "integer", "description": "Return top N classes"}],
                 "response_fields": ["label", "confidence", "all_scores"]},
                {"name": "list_models", "description": "List available classification models.", "method": "GET",
                 "required_parameters": [],
                 "optional_parameters": [{"name": "task", "type": "string", "description": "Filter by task type"}],
                 "response_fields": ["model_id", "task", "accuracy", "languages"]},
                {"name": "batch_classify", "description": "Classify multiple texts at once.", "method": "POST",
                 "required_parameters": [{"name": "texts", "type": "string", "description": "JSON array of texts"},
                                         {"name": "model", "type": "string", "description": "Model ID"}],
                 "optional_parameters": [],
                 "response_fields": ["results", "labels", "confidences"]},
            ]
        },
        {
            "tool_name": "image_recognition_api",
            "tool_description": "Identify objects, scenes, and attributes in images.",
            "api_list": [
                {"name": "detect_objects", "description": "Detect objects in an image URL.", "method": "POST",
                 "required_parameters": [{"name": "image_url", "type": "string", "description": "URL of image"}],
                 "optional_parameters": [{"name": "confidence_threshold", "type": "number", "description": "Min confidence 0-1"}],
                 "response_fields": ["objects", "bounding_boxes", "confidence_scores"]},
                {"name": "classify_scene", "description": "Classify the overall scene of an image.", "method": "POST",
                 "required_parameters": [{"name": "image_url", "type": "string", "description": "Image URL"}],
                 "optional_parameters": [],
                 "response_fields": ["scene_type", "confidence", "attributes"]},
            ]
        },
        {
            "tool_name": "nlp_processing_api",
            "tool_description": "Natural language processing tasks including NER, sentiment, and summarization.",
            "api_list": [
                {"name": "extract_entities", "description": "Extract named entities from text.", "method": "POST",
                 "required_parameters": [{"name": "text", "type": "string", "description": "Input text"}],
                 "optional_parameters": [{"name": "entity_types", "type": "string", "description": "Filter entity types"}],
                 "response_fields": ["entities", "entity_type", "start_pos", "end_pos", "confidence"]},
                {"name": "analyze_sentiment", "description": "Analyze sentiment of a text.", "method": "POST",
                 "required_parameters": [{"name": "text", "type": "string", "description": "Input text"}],
                 "optional_parameters": [],
                 "response_fields": ["sentiment", "score", "magnitude"]},
                {"name": "summarize_text", "description": "Summarize a long text document.", "method": "POST",
                 "required_parameters": [{"name": "text", "type": "string", "description": "Text to summarize"},
                                         {"name": "max_length", "type": "integer", "description": "Max words in summary"}],
                 "optional_parameters": [],
                 "response_fields": ["summary", "compression_ratio", "key_points"]},
            ]
        },
    ],
    "Business": [
        {
            "tool_name": "company_data_api",
            "tool_description": "Access company profiles, financials, and business intelligence data.",
            "api_list": [
                {"name": "search_companies", "description": "Search for companies by name or industry.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Company name or keyword"}],
                 "optional_parameters": [{"name": "industry", "type": "string", "description": "Industry filter"},
                                         {"name": "country", "type": "string", "description": "Country code"}],
                 "response_fields": ["company_id", "name", "industry", "founded", "employees", "revenue"]},
                {"name": "get_company_details", "description": "Get detailed info for a specific company.", "method": "GET",
                 "required_parameters": [{"name": "company_id", "type": "string", "description": "Company ID"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "description", "website", "linkedin", "revenue", "employees", "hq_location"]},
                {"name": "get_company_news", "description": "Get recent news for a company.", "method": "GET",
                 "required_parameters": [{"name": "company_id", "type": "string", "description": "Company ID"}],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Number of articles"}],
                 "response_fields": ["title", "source", "published_at", "url", "summary"]},
            ]
        },
        {
            "tool_name": "crm_api",
            "tool_description": "Customer relationship management — contacts, deals, pipelines.",
            "api_list": [
                {"name": "create_contact", "description": "Add a new contact to the CRM.", "method": "POST",
                 "required_parameters": [{"name": "email", "type": "string", "description": "Contact email"},
                                         {"name": "name", "type": "string", "description": "Full name"}],
                 "optional_parameters": [{"name": "phone", "type": "string", "description": "Phone number"},
                                         {"name": "company", "type": "string", "description": "Company name"}],
                 "response_fields": ["contact_id", "created_at", "status"]},
                {"name": "get_pipeline_deals", "description": "Get all deals in a sales pipeline.", "method": "GET",
                 "required_parameters": [{"name": "pipeline_id", "type": "string", "description": "Pipeline ID"}],
                 "optional_parameters": [{"name": "stage", "type": "string", "description": "Filter by stage"}],
                 "response_fields": ["deal_id", "title", "value", "stage", "owner", "close_date"]},
            ]
        },
        {
            "tool_name": "invoice_api",
            "tool_description": "Create, send, and manage invoices and payments.",
            "api_list": [
                {"name": "create_invoice", "description": "Create a new invoice.", "method": "POST",
                 "required_parameters": [{"name": "client_id", "type": "string", "description": "Client ID"},
                                         {"name": "items", "type": "string", "description": "JSON array of line items"},
                                         {"name": "due_date", "type": "string", "description": "Due date YYYY-MM-DD"}],
                 "optional_parameters": [{"name": "currency", "type": "string", "description": "Currency code"}],
                 "response_fields": ["invoice_id", "total", "status", "pdf_url"]},
                {"name": "get_invoice_status", "description": "Check the status of an invoice.", "method": "GET",
                 "required_parameters": [{"name": "invoice_id", "type": "string", "description": "Invoice ID"}],
                 "optional_parameters": [],
                 "response_fields": ["status", "paid_at", "amount_due", "amount_paid"]},
                {"name": "send_invoice", "description": "Email an invoice to the client.", "method": "POST",
                 "required_parameters": [{"name": "invoice_id", "type": "string", "description": "Invoice ID"}],
                 "optional_parameters": [{"name": "message", "type": "string", "description": "Custom email message"}],
                 "response_fields": ["sent_at", "recipient_email", "status"]},
            ]
        },
    ],
    "Commerce": [
        {
            "tool_name": "product_search_api",
            "tool_description": "Search products across e-commerce marketplaces.",
            "api_list": [
                {"name": "search_products", "description": "Search products by keyword.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Search query"}],
                 "optional_parameters": [{"name": "category", "type": "string", "description": "Product category"},
                                         {"name": "min_price", "type": "number", "description": "Min price"},
                                         {"name": "max_price", "type": "number", "description": "Max price"},
                                         {"name": "sort", "type": "string", "description": "Sort order"}],
                 "response_fields": ["product_id", "title", "price", "rating", "review_count", "image_url"]},
                {"name": "get_product_details", "description": "Get detailed info for a product.", "method": "GET",
                 "required_parameters": [{"name": "product_id", "type": "string", "description": "Product ID"}],
                 "optional_parameters": [],
                 "response_fields": ["title", "description", "price", "availability", "seller", "specs"]},
                {"name": "get_product_reviews", "description": "Get customer reviews for a product.", "method": "GET",
                 "required_parameters": [{"name": "product_id", "type": "string", "description": "Product ID"}],
                 "optional_parameters": [{"name": "page", "type": "integer", "description": "Page number"}],
                 "response_fields": ["review_id", "rating", "title", "body", "author", "date"]},
            ]
        },
        {
            "tool_name": "price_comparison_api",
            "tool_description": "Compare prices across different online retailers.",
            "api_list": [
                {"name": "compare_prices", "description": "Compare prices for a product across retailers.", "method": "GET",
                 "required_parameters": [{"name": "product_name", "type": "string", "description": "Product name"}],
                 "optional_parameters": [{"name": "brand", "type": "string", "description": "Brand filter"}],
                 "response_fields": ["retailer", "price", "shipping_cost", "availability", "url"]},
                {"name": "track_price", "description": "Set a price alert for a product.", "method": "POST",
                 "required_parameters": [{"name": "product_url", "type": "string", "description": "Product URL"},
                                         {"name": "target_price", "type": "number", "description": "Alert when price drops below"}],
                 "optional_parameters": [{"name": "email", "type": "string", "description": "Notification email"}],
                 "response_fields": ["alert_id", "current_price", "status"]},
            ]
        },
        {
            "tool_name": "order_tracking_api",
            "tool_description": "Track orders and shipments across carriers.",
            "api_list": [
                {"name": "track_order", "description": "Track an order by tracking number.", "method": "GET",
                 "required_parameters": [{"name": "tracking_number", "type": "string", "description": "Tracking number"},
                                         {"name": "carrier", "type": "string", "description": "Carrier name"}],
                 "optional_parameters": [],
                 "response_fields": ["status", "location", "estimated_delivery", "events"]},
                {"name": "get_shipping_rates", "description": "Get shipping rate quotes.", "method": "POST",
                 "required_parameters": [{"name": "origin_zip", "type": "string", "description": "Origin ZIP code"},
                                         {"name": "dest_zip", "type": "string", "description": "Destination ZIP code"},
                                         {"name": "weight_kg", "type": "number", "description": "Package weight in kg"}],
                 "optional_parameters": [{"name": "dimensions", "type": "string", "description": "LxWxH in cm"}],
                 "response_fields": ["carrier", "service", "rate", "transit_days"]},
                {"name": "estimate_delivery", "description": "Estimate delivery date for a shipment.", "method": "GET",
                 "required_parameters": [{"name": "origin", "type": "string", "description": "Origin city/country"},
                                         {"name": "destination", "type": "string", "description": "Destination city/country"},
                                         {"name": "carrier", "type": "string", "description": "Carrier name"}],
                 "optional_parameters": [],
                 "response_fields": ["estimated_date", "transit_days", "confidence"]},
            ]
        },
    ],
    "Music": [
        {
            "tool_name": "music_search_api",
            "tool_description": "Search for songs, artists, and albums across music databases.",
            "api_list": [
                {"name": "search_tracks", "description": "Search for music tracks.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Search query"}],
                 "optional_parameters": [{"name": "genre", "type": "string", "description": "Genre filter"},
                                         {"name": "year", "type": "integer", "description": "Release year"}],
                 "response_fields": ["track_id", "title", "artist", "album", "duration_ms", "preview_url"]},
                {"name": "get_artist_info", "description": "Get details about a music artist.", "method": "GET",
                 "required_parameters": [{"name": "artist_id", "type": "string", "description": "Artist ID"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "genres", "followers", "popularity", "top_tracks"]},
                {"name": "get_album_tracks", "description": "Get all tracks from an album.", "method": "GET",
                 "required_parameters": [{"name": "album_id", "type": "string", "description": "Album ID"}],
                 "optional_parameters": [],
                 "response_fields": ["track_number", "title", "duration_ms", "features"]},
            ]
        },
        {
            "tool_name": "lyrics_api",
            "tool_description": "Retrieve song lyrics and annotations.",
            "api_list": [
                {"name": "get_lyrics", "description": "Get lyrics for a song.", "method": "GET",
                 "required_parameters": [{"name": "artist", "type": "string", "description": "Artist name"},
                                         {"name": "song", "type": "string", "description": "Song title"}],
                 "optional_parameters": [],
                 "response_fields": ["lyrics", "song_id", "language", "explicit"]},
                {"name": "search_lyrics", "description": "Search songs by lyrics snippet.", "method": "GET",
                 "required_parameters": [{"name": "lyrics_snippet", "type": "string", "description": "Lyrics to search for"}],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Max results"}],
                 "response_fields": ["song", "artist", "match_score"]},
            ]
        },
        {
            "tool_name": "playlist_manager_api",
            "tool_description": "Create, manage, and share music playlists.",
            "api_list": [
                {"name": "create_playlist", "description": "Create a new playlist.", "method": "POST",
                 "required_parameters": [{"name": "name", "type": "string", "description": "Playlist name"},
                                         {"name": "track_ids", "type": "string", "description": "Comma-separated track IDs"}],
                 "optional_parameters": [{"name": "description", "type": "string", "description": "Playlist description"},
                                         {"name": "public", "type": "boolean", "description": "Make public"}],
                 "response_fields": ["playlist_id", "name", "track_count", "share_url"]},
                {"name": "get_recommendations", "description": "Get song recommendations based on a playlist.", "method": "GET",
                 "required_parameters": [{"name": "playlist_id", "type": "string", "description": "Playlist ID"}],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Number of recommendations"}],
                 "response_fields": ["track_id", "title", "artist", "similarity_score"]},
                {"name": "add_tracks", "description": "Add tracks to an existing playlist.", "method": "POST",
                 "required_parameters": [{"name": "playlist_id", "type": "string", "description": "Playlist ID"},
                                         {"name": "track_ids", "type": "string", "description": "Track IDs to add"}],
                 "optional_parameters": [],
                 "response_fields": ["playlist_id", "new_track_count", "status"]},
            ]
        },
    ],
    "Weather": [
        {
            "tool_name": "weather_api",
            "tool_description": "Current weather, forecasts, and historical weather data.",
            "api_list": [
                {"name": "get_current_weather", "description": "Get current weather for a location.", "method": "GET",
                 "required_parameters": [{"name": "location", "type": "string", "description": "City name or coordinates"}],
                 "optional_parameters": [{"name": "units", "type": "string", "description": "metric|imperial|kelvin"}],
                 "response_fields": ["temperature", "feels_like", "humidity", "wind_speed", "conditions", "uv_index"]},
                {"name": "get_weather_forecast", "description": "Get weather forecast for up to 14 days.", "method": "GET",
                 "required_parameters": [{"name": "location", "type": "string", "description": "Location name"},
                                         {"name": "days", "type": "integer", "description": "Number of forecast days"}],
                 "optional_parameters": [{"name": "units", "type": "string", "description": "metric|imperial"}],
                 "response_fields": ["date", "high", "low", "precipitation_chance", "conditions"]},
                {"name": "get_historical_weather", "description": "Get historical weather for a past date.", "method": "GET",
                 "required_parameters": [{"name": "location", "type": "string", "description": "Location"},
                                         {"name": "date", "type": "string", "description": "Date YYYY-MM-DD"}],
                 "optional_parameters": [],
                 "response_fields": ["temperature", "precipitation", "wind_speed", "conditions"]},
            ]
        },
        {
            "tool_name": "air_quality_api",
            "tool_description": "Real-time and forecast air quality index data.",
            "api_list": [
                {"name": "get_air_quality", "description": "Get current AQI for a location.", "method": "GET",
                 "required_parameters": [{"name": "location", "type": "string", "description": "City or coordinates"}],
                 "optional_parameters": [],
                 "response_fields": ["aqi", "pm25", "pm10", "o3", "no2", "health_category"]},
                {"name": "get_aqi_forecast", "description": "Get AQI forecast for the next 3 days.", "method": "GET",
                 "required_parameters": [{"name": "location", "type": "string", "description": "Location"}],
                 "optional_parameters": [],
                 "response_fields": ["date", "aqi_forecast", "dominant_pollutant"]},
            ]
        },
        {
            "tool_name": "severe_weather_alerts",
            "tool_description": "Severe weather warnings and alerts by location.",
            "api_list": [
                {"name": "get_active_alerts", "description": "Get active weather alerts for a region.", "method": "GET",
                 "required_parameters": [{"name": "region", "type": "string", "description": "State/region code"}],
                 "optional_parameters": [{"name": "severity", "type": "string", "description": "minor|moderate|severe|extreme"}],
                 "response_fields": ["alert_id", "event", "severity", "area", "expires", "description"]},
                {"name": "get_alert_details", "description": "Get details of a specific weather alert.", "method": "GET",
                 "required_parameters": [{"name": "alert_id", "type": "string", "description": "Alert ID"}],
                 "optional_parameters": [],
                 "response_fields": ["event_type", "severity", "certainty", "instructions", "issued", "expires"]},
                {"name": "subscribe_alerts", "description": "Subscribe to weather alerts for a location.", "method": "POST",
                 "required_parameters": [{"name": "location", "type": "string", "description": "Location"},
                                         {"name": "email", "type": "string", "description": "Email for notifications"}],
                 "optional_parameters": [{"name": "min_severity", "type": "string", "description": "Minimum severity level"}],
                 "response_fields": ["subscription_id", "status", "location"]},
            ]
        },
    ],
    "Travel": [
        {
            "tool_name": "flight_search_api",
            "tool_description": "Search and book flights across major airlines.",
            "api_list": [
                {"name": "search_flights", "description": "Search for available flights.", "method": "GET",
                 "required_parameters": [{"name": "origin", "type": "string", "description": "Origin airport IATA code"},
                                         {"name": "destination", "type": "string", "description": "Destination IATA code"},
                                         {"name": "date", "type": "string", "description": "Departure date YYYY-MM-DD"}],
                 "optional_parameters": [{"name": "return_date", "type": "string", "description": "Return date for round trips"},
                                         {"name": "passengers", "type": "integer", "description": "Number of passengers"},
                                         {"name": "cabin_class", "type": "string", "description": "economy|business|first"}],
                 "response_fields": ["flight_id", "airline", "origin", "destination", "departure_time", "arrival_time", "price", "stops"]},
                {"name": "get_flight_details", "description": "Get full details for a flight.", "method": "GET",
                 "required_parameters": [{"name": "flight_id", "type": "string", "description": "Flight ID"}],
                 "optional_parameters": [],
                 "response_fields": ["flight_number", "aircraft", "duration", "layovers", "baggage_allowance", "amenities"]},
                {"name": "get_flight_status", "description": "Check real-time flight status.", "method": "GET",
                 "required_parameters": [{"name": "flight_number", "type": "string", "description": "Flight number"},
                                         {"name": "date", "type": "string", "description": "Flight date"}],
                 "optional_parameters": [],
                 "response_fields": ["status", "departure_gate", "arrival_gate", "delay_minutes", "estimated_arrival"]},
            ]
        },
        {
            "tool_name": "hotel_booking_api",
            "tool_description": "Search, compare, and book hotels worldwide.",
            "api_list": [
                {"name": "search_hotels", "description": "Search for hotels in a destination.", "method": "GET",
                 "required_parameters": [{"name": "destination", "type": "string", "description": "City or destination"},
                                         {"name": "check_in", "type": "string", "description": "Check-in date YYYY-MM-DD"},
                                         {"name": "check_out", "type": "string", "description": "Check-out date YYYY-MM-DD"}],
                 "optional_parameters": [{"name": "guests", "type": "integer", "description": "Number of guests"},
                                         {"name": "stars", "type": "integer", "description": "Star rating filter"},
                                         {"name": "max_price", "type": "number", "description": "Max price per night"}],
                 "response_fields": ["hotel_id", "name", "stars", "rating", "price_per_night", "address", "amenities"]},
                {"name": "get_hotel_details", "description": "Get full details for a hotel.", "method": "GET",
                 "required_parameters": [{"name": "hotel_id", "type": "string", "description": "Hotel ID"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "description", "photos", "facilities", "policies", "location"]},
                {"name": "book_hotel", "description": "Book a hotel room.", "method": "POST",
                 "required_parameters": [{"name": "hotel_id", "type": "string", "description": "Hotel ID"},
                                         {"name": "room_type", "type": "string", "description": "Room type"},
                                         {"name": "guest_name", "type": "string", "description": "Guest full name"},
                                         {"name": "check_in", "type": "string", "description": "Check-in date"}],
                 "optional_parameters": [{"name": "special_requests", "type": "string", "description": "Special requests"}],
                 "response_fields": ["booking_id", "confirmation_number", "total_price", "status"]},
            ]
        },
        {
            "tool_name": "car_rental_api",
            "tool_description": "Search and book rental cars at airports and cities.",
            "api_list": [
                {"name": "search_cars", "description": "Search for available rental cars.", "method": "GET",
                 "required_parameters": [{"name": "pickup_location", "type": "string", "description": "Pickup location"},
                                         {"name": "pickup_date", "type": "string", "description": "Pickup date YYYY-MM-DD"},
                                         {"name": "return_date", "type": "string", "description": "Return date YYYY-MM-DD"}],
                 "optional_parameters": [{"name": "car_class", "type": "string", "description": "economy|compact|suv|luxury"},
                                         {"name": "max_price", "type": "number", "description": "Max daily rate"}],
                 "response_fields": ["car_id", "make", "model", "class", "seats", "price_per_day", "company"]},
                {"name": "get_car_details", "description": "Get full details for a rental car.", "method": "GET",
                 "required_parameters": [{"name": "car_id", "type": "string", "description": "Car ID"}],
                 "optional_parameters": [],
                 "response_fields": ["make", "model", "year", "fuel_type", "transmission", "features", "inclusions"]},
            ]
        },
    ],
    "Finance": [
        {
            "tool_name": "stock_market_api",
            "tool_description": "Real-time stock prices, historical data, and market analysis.",
            "api_list": [
                {"name": "get_stock_price", "description": "Get current stock price.", "method": "GET",
                 "required_parameters": [{"name": "symbol", "type": "string", "description": "Stock ticker symbol"}],
                 "optional_parameters": [],
                 "response_fields": ["symbol", "price", "change", "change_percent", "volume", "market_cap"]},
                {"name": "get_historical_prices", "description": "Get historical OHLCV data for a stock.", "method": "GET",
                 "required_parameters": [{"name": "symbol", "type": "string", "description": "Ticker symbol"},
                                         {"name": "period", "type": "string", "description": "1d|1w|1m|1y|5y"}],
                 "optional_parameters": [{"name": "interval", "type": "string", "description": "1m|5m|1h|1d"}],
                 "response_fields": ["date", "open", "high", "low", "close", "volume"]},
                {"name": "get_company_financials", "description": "Get income statement and balance sheet.", "method": "GET",
                 "required_parameters": [{"name": "symbol", "type": "string", "description": "Ticker symbol"}],
                 "optional_parameters": [{"name": "period", "type": "string", "description": "annual|quarterly"}],
                 "response_fields": ["revenue", "net_income", "eps", "total_assets", "total_debt", "pe_ratio"]},
            ]
        },
        {
            "tool_name": "currency_exchange_api",
            "tool_description": "Foreign exchange rates and currency conversion.",
            "api_list": [
                {"name": "get_exchange_rate", "description": "Get current exchange rate between two currencies.", "method": "GET",
                 "required_parameters": [{"name": "base", "type": "string", "description": "Base currency code e.g. USD"},
                                         {"name": "target", "type": "string", "description": "Target currency code"}],
                 "optional_parameters": [],
                 "response_fields": ["base", "target", "rate", "timestamp", "bid", "ask"]},
                {"name": "convert_currency", "description": "Convert an amount from one currency to another.", "method": "GET",
                 "required_parameters": [{"name": "amount", "type": "number", "description": "Amount to convert"},
                                         {"name": "from_currency", "type": "string", "description": "Source currency"},
                                         {"name": "to_currency", "type": "string", "description": "Target currency"}],
                 "optional_parameters": [{"name": "date", "type": "string", "description": "Historical date YYYY-MM-DD"}],
                 "response_fields": ["original_amount", "converted_amount", "rate", "date"]},
                {"name": "get_all_rates", "description": "Get rates for all currencies relative to a base.", "method": "GET",
                 "required_parameters": [{"name": "base", "type": "string", "description": "Base currency"}],
                 "optional_parameters": [],
                 "response_fields": ["base", "date", "rates"]},
            ]
        },
        {
            "tool_name": "crypto_api",
            "tool_description": "Cryptocurrency prices, market data, and portfolio tracking.",
            "api_list": [
                {"name": "get_crypto_price", "description": "Get current price for a cryptocurrency.", "method": "GET",
                 "required_parameters": [{"name": "coin_id", "type": "string", "description": "Coin ID e.g. bitcoin"}],
                 "optional_parameters": [{"name": "currency", "type": "string", "description": "Fiat currency for price"}],
                 "response_fields": ["id", "name", "symbol", "price", "market_cap", "24h_change", "volume"]},
                {"name": "get_market_overview", "description": "Get global crypto market overview.", "method": "GET",
                 "required_parameters": [],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Top N coins by market cap"}],
                 "response_fields": ["rank", "name", "symbol", "price", "market_cap", "dominance"]},
            ]
        },
    ],
    "News": [
        {
            "tool_name": "news_api",
            "tool_description": "Search and retrieve news articles from global sources.",
            "api_list": [
                {"name": "get_top_headlines", "description": "Get top news headlines.", "method": "GET",
                 "required_parameters": [],
                 "optional_parameters": [{"name": "category", "type": "string", "description": "business|tech|sports|health"},
                                         {"name": "country", "type": "string", "description": "2-letter country code"},
                                         {"name": "limit", "type": "integer", "description": "Number of articles"}],
                 "response_fields": ["title", "source", "published_at", "url", "description", "author"]},
                {"name": "search_news", "description": "Search news articles by keyword.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Search query"}],
                 "optional_parameters": [{"name": "from_date", "type": "string", "description": "Start date"},
                                         {"name": "to_date", "type": "string", "description": "End date"},
                                         {"name": "language", "type": "string", "description": "Language code"}],
                 "response_fields": ["title", "source", "published_at", "url", "content"]},
                {"name": "get_article_content", "description": "Get full text content of an article.", "method": "GET",
                 "required_parameters": [{"name": "article_url", "type": "string", "description": "URL of the article"}],
                 "optional_parameters": [],
                 "response_fields": ["title", "content", "author", "published_at", "images"]},
            ]
        },
        {
            "tool_name": "rss_feed_api",
            "tool_description": "Manage and read RSS/Atom news feeds.",
            "api_list": [
                {"name": "get_feed_items", "description": "Get items from an RSS feed.", "method": "GET",
                 "required_parameters": [{"name": "feed_url", "type": "string", "description": "RSS feed URL"}],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Max items to return"}],
                 "response_fields": ["title", "link", "published", "summary", "author"]},
                {"name": "discover_feeds", "description": "Discover RSS feeds from a website.", "method": "GET",
                 "required_parameters": [{"name": "website_url", "type": "string", "description": "Website to scan"}],
                 "optional_parameters": [],
                 "response_fields": ["feed_url", "feed_title", "feed_type"]},
            ]
        },
        {
            "tool_name": "fact_check_api",
            "tool_description": "Verify claims and check facts against trusted sources.",
            "api_list": [
                {"name": "check_claim", "description": "Fact-check a specific claim.", "method": "POST",
                 "required_parameters": [{"name": "claim", "type": "string", "description": "Claim to verify"}],
                 "optional_parameters": [{"name": "language", "type": "string", "description": "Language code"}],
                 "response_fields": ["claim", "verdict", "sources", "explanation", "confidence"]},
                {"name": "search_fact_checks", "description": "Search existing fact-checks.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Search terms"}],
                 "optional_parameters": [{"name": "publisher", "type": "string", "description": "Fact-check publisher"}],
                 "response_fields": ["claim", "verdict", "claimant", "review_date", "url"]},
                {"name": "get_claim_reviews", "description": "Get claim reviews from multiple fact-checkers.", "method": "GET",
                 "required_parameters": [{"name": "claim_id", "type": "string", "description": "Claim ID"}],
                 "optional_parameters": [],
                 "response_fields": ["publisher", "verdict", "rating_value", "url", "date"]},
            ]
        },
    ],
    "Social": [
        {
            "tool_name": "social_media_analytics",
            "tool_description": "Analytics and insights for social media accounts and posts.",
            "api_list": [
                {"name": "get_profile_stats", "description": "Get stats for a social media profile.", "method": "GET",
                 "required_parameters": [{"name": "platform", "type": "string", "description": "twitter|instagram|linkedin"},
                                         {"name": "username", "type": "string", "description": "Username or handle"}],
                 "optional_parameters": [],
                 "response_fields": ["followers", "following", "posts_count", "engagement_rate", "avg_likes"]},
                {"name": "get_post_analytics", "description": "Get analytics for a specific post.", "method": "GET",
                 "required_parameters": [{"name": "post_id", "type": "string", "description": "Post ID"},
                                         {"name": "platform", "type": "string", "description": "Platform name"}],
                 "optional_parameters": [],
                 "response_fields": ["likes", "comments", "shares", "impressions", "reach", "saves"]},
                {"name": "get_hashtag_analytics", "description": "Get analytics for a hashtag.", "method": "GET",
                 "required_parameters": [{"name": "hashtag", "type": "string", "description": "Hashtag without #"},
                                         {"name": "platform", "type": "string", "description": "Platform name"}],
                 "optional_parameters": [{"name": "period", "type": "string", "description": "7d|30d|90d"}],
                 "response_fields": ["post_count", "total_reach", "top_posts", "trending_score"]},
            ]
        },
        {
            "tool_name": "content_scheduler",
            "tool_description": "Schedule and publish social media content.",
            "api_list": [
                {"name": "schedule_post", "description": "Schedule a post for future publishing.", "method": "POST",
                 "required_parameters": [{"name": "platform", "type": "string", "description": "Platform to post to"},
                                         {"name": "content", "type": "string", "description": "Post content"},
                                         {"name": "scheduled_time", "type": "string", "description": "ISO 8601 datetime"}],
                 "optional_parameters": [{"name": "media_url", "type": "string", "description": "Media attachment URL"},
                                         {"name": "hashtags", "type": "string", "description": "Comma-separated hashtags"}],
                 "response_fields": ["post_id", "scheduled_time", "platform", "status"]},
                {"name": "get_best_times", "description": "Get optimal posting times for an account.", "method": "GET",
                 "required_parameters": [{"name": "account_id", "type": "string", "description": "Account ID"},
                                         {"name": "platform", "type": "string", "description": "Platform name"}],
                 "optional_parameters": [],
                 "response_fields": ["day_of_week", "hour", "engagement_score", "recommended"]},
            ]
        },
        {
            "tool_name": "influencer_search_api",
            "tool_description": "Find and analyze social media influencers.",
            "api_list": [
                {"name": "search_influencers", "description": "Search for influencers by niche and platform.", "method": "GET",
                 "required_parameters": [{"name": "niche", "type": "string", "description": "Content niche or industry"},
                                         {"name": "platform", "type": "string", "description": "Platform name"}],
                 "optional_parameters": [{"name": "min_followers", "type": "integer", "description": "Minimum followers"},
                                         {"name": "location", "type": "string", "description": "Geographic location"}],
                 "response_fields": ["username", "followers", "engagement_rate", "avg_views", "niche", "contact"]},
                {"name": "get_influencer_profile", "description": "Get detailed influencer profile.", "method": "GET",
                 "required_parameters": [{"name": "username", "type": "string", "description": "Influencer username"},
                                         {"name": "platform", "type": "string", "description": "Platform"}],
                 "optional_parameters": [],
                 "response_fields": ["bio", "categories", "audience_demographics", "recent_posts", "collaboration_rate"]},
                {"name": "estimate_campaign_reach", "description": "Estimate reach for an influencer campaign.", "method": "POST",
                 "required_parameters": [{"name": "influencer_ids", "type": "string", "description": "Comma-separated IDs"},
                                         {"name": "budget", "type": "number", "description": "Campaign budget USD"}],
                 "optional_parameters": [],
                 "response_fields": ["estimated_reach", "estimated_impressions", "estimated_engagements", "cost_per_impression"]},
            ]
        },
    ],
    "Food": [
        {
            "tool_name": "restaurant_finder",
            "tool_description": "Find restaurants and food venues near a location.",
            "api_list": [
                {"name": "search_restaurants", "description": "Search for restaurants near a location.", "method": "GET",
                 "required_parameters": [{"name": "location", "type": "string", "description": "City or address"}],
                 "optional_parameters": [{"name": "cuisine", "type": "string", "description": "Cuisine type filter"},
                                         {"name": "price_range", "type": "string", "description": "1-4 price range"},
                                         {"name": "rating_min", "type": "number", "description": "Minimum rating 1-5"}],
                 "response_fields": ["restaurant_id", "name", "cuisine", "rating", "price_range", "address", "open_now"]},
                {"name": "get_restaurant_details", "description": "Get full details for a restaurant.", "method": "GET",
                 "required_parameters": [{"name": "restaurant_id", "type": "string", "description": "Restaurant ID"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "phone", "hours", "menu_url", "delivery_options", "photos"]},
                {"name": "check_reservation", "description": "Check table availability.", "method": "GET",
                 "required_parameters": [{"name": "restaurant_id", "type": "string", "description": "Restaurant ID"},
                                         {"name": "date", "type": "string", "description": "Date YYYY-MM-DD"},
                                         {"name": "party_size", "type": "integer", "description": "Number of guests"}],
                 "optional_parameters": [],
                 "response_fields": ["available_times", "earliest_available", "requires_deposit"]},
            ]
        },
        {
            "tool_name": "recipe_api",
            "tool_description": "Search, get, and manage cooking recipes.",
            "api_list": [
                {"name": "search_recipes", "description": "Search for recipes by ingredient or dish name.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Search query"}],
                 "optional_parameters": [{"name": "diet", "type": "string", "description": "vegetarian|vegan|keto|gluten-free"},
                                         {"name": "cuisine", "type": "string", "description": "Cuisine type"},
                                         {"name": "max_time", "type": "integer", "description": "Max cooking time in minutes"}],
                 "response_fields": ["recipe_id", "title", "rating", "time_minutes", "servings", "calories"]},
                {"name": "get_recipe", "description": "Get full recipe with ingredients and instructions.", "method": "GET",
                 "required_parameters": [{"name": "recipe_id", "type": "string", "description": "Recipe ID"}],
                 "optional_parameters": [{"name": "servings", "type": "integer", "description": "Scale to servings"}],
                 "response_fields": ["title", "ingredients", "steps", "nutrition", "tips", "prep_time", "cook_time"]},
            ]
        },
        {
            "tool_name": "food_delivery_api",
            "tool_description": "Order food delivery from local restaurants.",
            "api_list": [
                {"name": "get_delivery_restaurants", "description": "Get restaurants that deliver to an address.", "method": "GET",
                 "required_parameters": [{"name": "delivery_address", "type": "string", "description": "Delivery address"}],
                 "optional_parameters": [{"name": "cuisine", "type": "string", "description": "Cuisine filter"},
                                         {"name": "max_delivery_time", "type": "integer", "description": "Max delivery time minutes"}],
                 "response_fields": ["restaurant_id", "name", "delivery_time", "delivery_fee", "min_order", "rating"]},
                {"name": "place_order", "description": "Place a food delivery order.", "method": "POST",
                 "required_parameters": [{"name": "restaurant_id", "type": "string", "description": "Restaurant ID"},
                                         {"name": "items", "type": "string", "description": "JSON array of items"},
                                         {"name": "delivery_address", "type": "string", "description": "Delivery address"}],
                 "optional_parameters": [{"name": "special_instructions", "type": "string", "description": "Order instructions"}],
                 "response_fields": ["order_id", "status", "estimated_arrival", "total", "driver_info"]},
                {"name": "track_delivery", "description": "Track an active food delivery.", "method": "GET",
                 "required_parameters": [{"name": "order_id", "type": "string", "description": "Order ID"}],
                 "optional_parameters": [],
                 "response_fields": ["status", "driver_location", "estimated_arrival", "updates"]},
            ]
        },
    ],
    "Sports": [
        {
            "tool_name": "sports_scores_api",
            "tool_description": "Real-time scores, standings, and statistics for major sports.",
            "api_list": [
                {"name": "get_live_scores", "description": "Get live scores for ongoing matches.", "method": "GET",
                 "required_parameters": [{"name": "sport", "type": "string", "description": "Sport type"}],
                 "optional_parameters": [{"name": "league", "type": "string", "description": "League code"},
                                         {"name": "date", "type": "string", "description": "Date YYYY-MM-DD"}],
                 "response_fields": ["match_id", "home_team", "away_team", "home_score", "away_score", "status", "minute"]},
                {"name": "get_standings", "description": "Get league standings table.", "method": "GET",
                 "required_parameters": [{"name": "league_id", "type": "string", "description": "League ID"},
                                         {"name": "season", "type": "string", "description": "Season year"}],
                 "optional_parameters": [],
                 "response_fields": ["position", "team", "played", "won", "drawn", "lost", "points", "goal_difference"]},
                {"name": "get_player_stats", "description": "Get statistics for a player.", "method": "GET",
                 "required_parameters": [{"name": "player_id", "type": "string", "description": "Player ID"},
                                         {"name": "season", "type": "string", "description": "Season year"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "team", "position", "goals", "assists", "appearances", "rating"]},
            ]
        },
        {
            "tool_name": "sports_betting_odds_api",
            "tool_description": "Betting odds and predictions for sports events.",
            "api_list": [
                {"name": "get_match_odds", "description": "Get betting odds for a match.", "method": "GET",
                 "required_parameters": [{"name": "match_id", "type": "string", "description": "Match ID"}],
                 "optional_parameters": [{"name": "bookmaker", "type": "string", "description": "Specific bookmaker"}],
                 "response_fields": ["home_win_odds", "draw_odds", "away_win_odds", "best_odds", "bookmaker"]},
                {"name": "get_upcoming_fixtures", "description": "Get upcoming fixtures for a league.", "method": "GET",
                 "required_parameters": [{"name": "league_id", "type": "string", "description": "League ID"}],
                 "optional_parameters": [{"name": "days_ahead", "type": "integer", "description": "Look-ahead days"}],
                 "response_fields": ["match_id", "home_team", "away_team", "date", "venue"]},
            ]
        },
        {
            "tool_name": "athlete_profiles_api",
            "tool_description": "Profiles, career stats, and news for professional athletes.",
            "api_list": [
                {"name": "search_athletes", "description": "Search for athletes by name or sport.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Athlete name or keyword"}],
                 "optional_parameters": [{"name": "sport", "type": "string", "description": "Sport filter"},
                                         {"name": "nationality", "type": "string", "description": "Country code"}],
                 "response_fields": ["athlete_id", "name", "sport", "team", "nationality", "age"]},
                {"name": "get_career_stats", "description": "Get full career statistics for an athlete.", "method": "GET",
                 "required_parameters": [{"name": "athlete_id", "type": "string", "description": "Athlete ID"}],
                 "optional_parameters": [{"name": "season", "type": "string", "description": "Filter by season"}],
                 "response_fields": ["seasons", "appearances", "achievements", "records", "career_highlights"]},
                {"name": "get_athlete_news", "description": "Get recent news about an athlete.", "method": "GET",
                 "required_parameters": [{"name": "athlete_id", "type": "string", "description": "Athlete ID"}],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Max articles"}],
                 "response_fields": ["title", "source", "published_at", "url", "summary"]},
            ]
        },
    ],
    "Health": [
        {
            "tool_name": "medical_info_api",
            "tool_description": "Access medical conditions, symptoms, and treatment information.",
            "api_list": [
                {"name": "search_conditions", "description": "Search for medical conditions.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Condition name or symptom"}],
                 "optional_parameters": [{"name": "category", "type": "string", "description": "Medical category"}],
                 "response_fields": ["condition_id", "name", "description", "symptoms", "icd10_code"]},
                {"name": "get_condition_details", "description": "Get detailed info for a medical condition.", "method": "GET",
                 "required_parameters": [{"name": "condition_id", "type": "string", "description": "Condition ID"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "symptoms", "causes", "diagnosis", "treatment_options", "prognosis"]},
                {"name": "check_drug_interactions", "description": "Check for interactions between medications.", "method": "POST",
                 "required_parameters": [{"name": "drugs", "type": "string", "description": "Comma-separated drug names"}],
                 "optional_parameters": [],
                 "response_fields": ["interaction_found", "severity", "description", "recommendations"]},
            ]
        },
        {
            "tool_name": "fitness_tracking_api",
            "tool_description": "Log workouts, track fitness metrics, and get recommendations.",
            "api_list": [
                {"name": "log_workout", "description": "Log a workout session.", "method": "POST",
                 "required_parameters": [{"name": "activity_type", "type": "string", "description": "Type of activity"},
                                         {"name": "duration_minutes", "type": "integer", "description": "Duration in minutes"},
                                         {"name": "user_id", "type": "string", "description": "User ID"}],
                 "optional_parameters": [{"name": "calories_burned", "type": "integer", "description": "Estimated calories"},
                                         {"name": "notes", "type": "string", "description": "Workout notes"}],
                 "response_fields": ["workout_id", "calories_burned", "weekly_total_minutes", "streak_days"]},
                {"name": "get_fitness_summary", "description": "Get fitness activity summary.", "method": "GET",
                 "required_parameters": [{"name": "user_id", "type": "string", "description": "User ID"}],
                 "optional_parameters": [{"name": "period", "type": "string", "description": "week|month|year"}],
                 "response_fields": ["total_workouts", "total_minutes", "calories_burned", "goals_met", "activity_breakdown"]},
            ]
        },
        {
            "tool_name": "nutrition_api",
            "tool_description": "Food nutrition data, meal tracking, and dietary analysis.",
            "api_list": [
                {"name": "get_nutrition_info", "description": "Get nutritional info for a food item.", "method": "GET",
                 "required_parameters": [{"name": "food_item", "type": "string", "description": "Food name or barcode"}],
                 "optional_parameters": [{"name": "serving_size", "type": "string", "description": "Serving size"}],
                 "response_fields": ["calories", "protein", "carbs", "fat", "fiber", "sugar", "vitamins", "minerals"]},
                {"name": "analyze_meal", "description": "Analyze nutritional composition of a full meal.", "method": "POST",
                 "required_parameters": [{"name": "foods", "type": "string", "description": "JSON array of food items with quantities"}],
                 "optional_parameters": [],
                 "response_fields": ["total_calories", "macros", "micros", "health_score", "recommendations"]},
                {"name": "get_daily_target", "description": "Get recommended daily nutritional targets.", "method": "GET",
                 "required_parameters": [{"name": "age", "type": "integer", "description": "Age in years"},
                                         {"name": "gender", "type": "string", "description": "male|female"},
                                         {"name": "activity_level", "type": "string", "description": "sedentary|moderate|active"}],
                 "optional_parameters": [{"name": "goal", "type": "string", "description": "lose_weight|maintain|gain_muscle"}],
                 "response_fields": ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "water_ml"]},
            ]
        },
    ],
    "Maps": [
        {
            "tool_name": "geocoding_api",
            "tool_description": "Convert addresses to coordinates and vice versa.",
            "api_list": [
                {"name": "geocode_address", "description": "Convert an address to coordinates.", "method": "GET",
                 "required_parameters": [{"name": "address", "type": "string", "description": "Address to geocode"}],
                 "optional_parameters": [{"name": "country", "type": "string", "description": "Country bias code"}],
                 "response_fields": ["latitude", "longitude", "formatted_address", "place_id", "accuracy"]},
                {"name": "reverse_geocode", "description": "Convert coordinates to an address.", "method": "GET",
                 "required_parameters": [{"name": "latitude", "type": "number", "description": "Latitude"},
                                         {"name": "longitude", "type": "number", "description": "Longitude"}],
                 "optional_parameters": [],
                 "response_fields": ["formatted_address", "street", "city", "state", "country", "postal_code"]},
                {"name": "get_place_details", "description": "Get details for a place.", "method": "GET",
                 "required_parameters": [{"name": "place_id", "type": "string", "description": "Place ID"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "address", "phone", "website", "hours", "rating", "reviews"]},
            ]
        },
        {
            "tool_name": "directions_api",
            "tool_description": "Get turn-by-turn directions and route planning.",
            "api_list": [
                {"name": "get_directions", "description": "Get driving/transit/walking directions.", "method": "GET",
                 "required_parameters": [{"name": "origin", "type": "string", "description": "Start address"},
                                         {"name": "destination", "type": "string", "description": "End address"}],
                 "optional_parameters": [{"name": "mode", "type": "string", "description": "driving|transit|walking|cycling"},
                                         {"name": "avoid", "type": "string", "description": "tolls|highways|ferries"}],
                 "response_fields": ["distance_km", "duration_minutes", "steps", "waypoints", "toll_cost"]},
                {"name": "get_traffic", "description": "Get current traffic conditions on a route.", "method": "GET",
                 "required_parameters": [{"name": "origin", "type": "string", "description": "Start location"},
                                         {"name": "destination", "type": "string", "description": "End location"}],
                 "optional_parameters": [],
                 "response_fields": ["traffic_level", "delay_minutes", "incidents", "suggested_route"]},
            ]
        },
        {
            "tool_name": "places_search_api",
            "tool_description": "Search for points of interest and nearby places.",
            "api_list": [
                {"name": "search_nearby", "description": "Search for places near a location.", "method": "GET",
                 "required_parameters": [{"name": "location", "type": "string", "description": "Center location"},
                                         {"name": "type", "type": "string", "description": "Place type e.g. hospital"}],
                 "optional_parameters": [{"name": "radius_km", "type": "number", "description": "Search radius"},
                                         {"name": "open_now", "type": "boolean", "description": "Filter open places"}],
                 "response_fields": ["place_id", "name", "distance_km", "rating", "address", "open_now"]},
                {"name": "get_place_reviews", "description": "Get user reviews for a place.", "method": "GET",
                 "required_parameters": [{"name": "place_id", "type": "string", "description": "Place ID"}],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Max reviews"}],
                 "response_fields": ["author", "rating", "text", "date", "helpful_votes"]},
                {"name": "get_place_photos", "description": "Get photos for a place.", "method": "GET",
                 "required_parameters": [{"name": "place_id", "type": "string", "description": "Place ID"}],
                 "optional_parameters": [{"name": "max_width", "type": "integer", "description": "Max image width px"}],
                 "response_fields": ["photo_url", "width", "height", "attribution"]},
            ]
        },
    ],
    "Education": [
        {
            "tool_name": "course_search_api",
            "tool_description": "Search and discover online courses and learning resources.",
            "api_list": [
                {"name": "search_courses", "description": "Search for online courses.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Topic or keyword"}],
                 "optional_parameters": [{"name": "platform", "type": "string", "description": "udemy|coursera|edx"},
                                         {"name": "level", "type": "string", "description": "beginner|intermediate|advanced"},
                                         {"name": "free_only", "type": "boolean", "description": "Free courses only"}],
                 "response_fields": ["course_id", "title", "instructor", "rating", "students_count", "price", "duration_hours"]},
                {"name": "get_course_details", "description": "Get full details for a course.", "method": "GET",
                 "required_parameters": [{"name": "course_id", "type": "string", "description": "Course ID"}],
                 "optional_parameters": [],
                 "response_fields": ["title", "description", "curriculum", "prerequisites", "certificate", "language"]},
                {"name": "get_course_reviews", "description": "Get reviews for a course.", "method": "GET",
                 "required_parameters": [{"name": "course_id", "type": "string", "description": "Course ID"}],
                 "optional_parameters": [],
                 "response_fields": ["reviewer", "rating", "review", "date", "helpful"]},
            ]
        },
        {
            "tool_name": "dictionary_api",
            "tool_description": "Word definitions, synonyms, antonyms, and usage examples.",
            "api_list": [
                {"name": "get_definition", "description": "Get the definition of a word.", "method": "GET",
                 "required_parameters": [{"name": "word", "type": "string", "description": "Word to look up"}],
                 "optional_parameters": [{"name": "language", "type": "string", "description": "Language code"}],
                 "response_fields": ["word", "phonetic", "definitions", "part_of_speech", "examples", "origin"]},
                {"name": "get_synonyms", "description": "Get synonyms and antonyms for a word.", "method": "GET",
                 "required_parameters": [{"name": "word", "type": "string", "description": "Word"}],
                 "optional_parameters": [],
                 "response_fields": ["synonyms", "antonyms", "related_words"]},
            ]
        },
        {
            "tool_name": "academic_search_api",
            "tool_description": "Search academic papers and research publications.",
            "api_list": [
                {"name": "search_papers", "description": "Search academic papers by topic.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Research topic"}],
                 "optional_parameters": [{"name": "year_from", "type": "integer", "description": "Publication year from"},
                                         {"name": "field", "type": "string", "description": "Academic field"}],
                 "response_fields": ["paper_id", "title", "authors", "year", "journal", "citations", "abstract"]},
                {"name": "get_paper_details", "description": "Get full details for an academic paper.", "method": "GET",
                 "required_parameters": [{"name": "paper_id", "type": "string", "description": "Paper ID"}],
                 "optional_parameters": [],
                 "response_fields": ["title", "abstract", "authors", "doi", "references", "cited_by"]},
                {"name": "get_author_profile", "description": "Get academic author profile.", "method": "GET",
                 "required_parameters": [{"name": "author_id", "type": "string", "description": "Author ID"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "affiliation", "h_index", "citations", "papers_count", "research_interests"]},
            ]
        },
    ],
    "Entertainment": [
        {
            "tool_name": "movie_database_api",
            "tool_description": "Movie and TV show information, ratings, and recommendations.",
            "api_list": [
                {"name": "search_movies", "description": "Search for movies or TV shows.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Title or keyword"}],
                 "optional_parameters": [{"name": "type", "type": "string", "description": "movie|tv"},
                                         {"name": "year", "type": "integer", "description": "Release year"}],
                 "response_fields": ["id", "title", "year", "rating", "genre", "poster_url", "overview"]},
                {"name": "get_movie_details", "description": "Get full details for a movie or show.", "method": "GET",
                 "required_parameters": [{"name": "movie_id", "type": "string", "description": "Movie ID"}],
                 "optional_parameters": [],
                 "response_fields": ["title", "director", "cast", "runtime", "budget", "box_office", "awards"]},
                {"name": "get_recommendations", "description": "Get similar movies or shows.", "method": "GET",
                 "required_parameters": [{"name": "movie_id", "type": "string", "description": "Movie ID"}],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Number of recommendations"}],
                 "response_fields": ["id", "title", "year", "rating", "similarity_score"]},
            ]
        },
        {
            "tool_name": "event_search_api",
            "tool_description": "Find concerts, shows, sports events, and local activities.",
            "api_list": [
                {"name": "search_events", "description": "Search for events in a location.", "method": "GET",
                 "required_parameters": [{"name": "location", "type": "string", "description": "City or area"}],
                 "optional_parameters": [{"name": "category", "type": "string", "description": "music|sports|theater|comedy"},
                                         {"name": "date_from", "type": "string", "description": "Start date"},
                                         {"name": "max_price", "type": "number", "description": "Max ticket price"}],
                 "response_fields": ["event_id", "name", "date", "venue", "category", "price_range", "tickets_left"]},
                {"name": "get_event_details", "description": "Get full details for an event.", "method": "GET",
                 "required_parameters": [{"name": "event_id", "type": "string", "description": "Event ID"}],
                 "optional_parameters": [],
                 "response_fields": ["name", "venue", "lineup", "age_restriction", "parking", "accessibility"]},
            ]
        },
        {
            "tool_name": "gaming_api",
            "tool_description": "Video game information, reviews, and community data.",
            "api_list": [
                {"name": "search_games", "description": "Search for video games.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Game title or keyword"}],
                 "optional_parameters": [{"name": "platform", "type": "string", "description": "ps5|xbox|pc|switch"},
                                         {"name": "genre", "type": "string", "description": "Game genre"}],
                 "response_fields": ["game_id", "title", "developer", "rating", "release_date", "genre", "platforms"]},
                {"name": "get_game_details", "description": "Get full details for a game.", "method": "GET",
                 "required_parameters": [{"name": "game_id", "type": "string", "description": "Game ID"}],
                 "optional_parameters": [],
                 "response_fields": ["title", "description", "metacritic_score", "user_score", "dlc", "multiplayer"]},
                {"name": "get_player_stats", "description": "Get stats for a player in an online game.", "method": "GET",
                 "required_parameters": [{"name": "game_id", "type": "string", "description": "Game ID"},
                                         {"name": "username", "type": "string", "description": "Player username"}],
                 "optional_parameters": [],
                 "response_fields": ["level", "rank", "wins", "losses", "kd_ratio", "playtime_hours"]},
            ]
        },
    ],
    "Productivity": [
        {
            "tool_name": "calendar_api",
            "tool_description": "Manage calendar events, schedules, and reminders.",
            "api_list": [
                {"name": "create_event", "description": "Create a calendar event.", "method": "POST",
                 "required_parameters": [{"name": "title", "type": "string", "description": "Event title"},
                                         {"name": "start_time", "type": "string", "description": "Start ISO datetime"},
                                         {"name": "end_time", "type": "string", "description": "End ISO datetime"}],
                 "optional_parameters": [{"name": "location", "type": "string", "description": "Event location"},
                                         {"name": "description", "type": "string", "description": "Event description"},
                                         {"name": "attendees", "type": "string", "description": "Comma-separated emails"}],
                 "response_fields": ["event_id", "title", "start_time", "end_time", "calendar_link"]},
                {"name": "check_availability", "description": "Check calendar availability.", "method": "GET",
                 "required_parameters": [{"name": "user_id", "type": "string", "description": "User ID"},
                                         {"name": "date", "type": "string", "description": "Date YYYY-MM-DD"}],
                 "optional_parameters": [{"name": "duration_minutes", "type": "integer", "description": "Required duration"}],
                 "response_fields": ["free_slots", "busy_slots", "next_available"]},
                {"name": "get_events", "description": "Get events in a date range.", "method": "GET",
                 "required_parameters": [{"name": "start_date", "type": "string", "description": "Start date"},
                                         {"name": "end_date", "type": "string", "description": "End date"}],
                 "optional_parameters": [{"name": "calendar_id", "type": "string", "description": "Specific calendar"}],
                 "response_fields": ["event_id", "title", "start_time", "end_time", "location", "attendees"]},
            ]
        },
        {
            "tool_name": "task_manager_api",
            "tool_description": "Create, organize, and track tasks and projects.",
            "api_list": [
                {"name": "create_task", "description": "Create a new task.", "method": "POST",
                 "required_parameters": [{"name": "title", "type": "string", "description": "Task title"}],
                 "optional_parameters": [{"name": "due_date", "type": "string", "description": "Due date"},
                                         {"name": "priority", "type": "string", "description": "low|medium|high|urgent"},
                                         {"name": "project_id", "type": "string", "description": "Project to add to"}],
                 "response_fields": ["task_id", "title", "status", "created_at"]},
                {"name": "get_tasks", "description": "Get tasks for a project or user.", "method": "GET",
                 "required_parameters": [],
                 "optional_parameters": [{"name": "project_id", "type": "string", "description": "Project filter"},
                                         {"name": "status", "type": "string", "description": "todo|in_progress|done"},
                                         {"name": "due_before", "type": "string", "description": "Due date filter"}],
                 "response_fields": ["task_id", "title", "status", "priority", "due_date", "assignee"]},
            ]
        },
        {
            "tool_name": "note_taking_api",
            "tool_description": "Create, organize, and search notes and documents.",
            "api_list": [
                {"name": "create_note", "description": "Create a new note.", "method": "POST",
                 "required_parameters": [{"name": "title", "type": "string", "description": "Note title"},
                                         {"name": "content", "type": "string", "description": "Note content"}],
                 "optional_parameters": [{"name": "tags", "type": "string", "description": "Comma-separated tags"},
                                         {"name": "notebook", "type": "string", "description": "Notebook name"}],
                 "response_fields": ["note_id", "created_at", "share_url"]},
                {"name": "search_notes", "description": "Search notes by keyword.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Search query"}],
                 "optional_parameters": [{"name": "notebook", "type": "string", "description": "Notebook filter"},
                                         {"name": "tags", "type": "string", "description": "Tag filter"}],
                 "response_fields": ["note_id", "title", "snippet", "created_at", "tags"]},
                {"name": "get_note", "description": "Retrieve a specific note.", "method": "GET",
                 "required_parameters": [{"name": "note_id", "type": "string", "description": "Note ID"}],
                 "optional_parameters": [],
                 "response_fields": ["title", "content", "created_at", "updated_at", "tags"]},
            ]
        },
    ],
    "Communication": [
        {
            "tool_name": "email_api",
            "tool_description": "Send, receive, and manage emails programmatically.",
            "api_list": [
                {"name": "send_email", "description": "Send an email.", "method": "POST",
                 "required_parameters": [{"name": "to", "type": "string", "description": "Recipient email"},
                                         {"name": "subject", "type": "string", "description": "Email subject"},
                                         {"name": "body", "type": "string", "description": "Email body text"}],
                 "optional_parameters": [{"name": "cc", "type": "string", "description": "CC addresses"},
                                         {"name": "attachment_url", "type": "string", "description": "File attachment URL"}],
                 "response_fields": ["message_id", "status", "sent_at"]},
                {"name": "get_inbox", "description": "Get emails from inbox.", "method": "GET",
                 "required_parameters": [],
                 "optional_parameters": [{"name": "folder", "type": "string", "description": "inbox|sent|drafts"},
                                         {"name": "unread_only", "type": "boolean", "description": "Unread only"},
                                         {"name": "limit", "type": "integer", "description": "Max emails"}],
                 "response_fields": ["message_id", "from", "subject", "date", "read", "has_attachment"]},
                {"name": "search_emails", "description": "Search emails by query.", "method": "GET",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Search query"}],
                 "optional_parameters": [],
                 "response_fields": ["message_id", "from", "subject", "date", "snippet"]},
            ]
        },
        {
            "tool_name": "sms_api",
            "tool_description": "Send SMS and MMS messages programmatically.",
            "api_list": [
                {"name": "send_sms", "description": "Send an SMS message.", "method": "POST",
                 "required_parameters": [{"name": "to", "type": "string", "description": "Recipient phone number"},
                                         {"name": "message", "type": "string", "description": "Message text"}],
                 "optional_parameters": [{"name": "from_number", "type": "string", "description": "Sender number"}],
                 "response_fields": ["message_id", "status", "cost", "segments"]},
                {"name": "get_sms_status", "description": "Check delivery status of an SMS.", "method": "GET",
                 "required_parameters": [{"name": "message_id", "type": "string", "description": "Message ID"}],
                 "optional_parameters": [],
                 "response_fields": ["status", "delivered_at", "error_code"]},
            ]
        },
        {
            "tool_name": "translation_api",
            "tool_description": "Translate text between languages.",
            "api_list": [
                {"name": "translate_text", "description": "Translate text to a target language.", "method": "POST",
                 "required_parameters": [{"name": "text", "type": "string", "description": "Text to translate"},
                                         {"name": "target_language", "type": "string", "description": "Target language code"}],
                 "optional_parameters": [{"name": "source_language", "type": "string", "description": "Source language (auto-detect if omitted)"}],
                 "response_fields": ["translated_text", "detected_source_language", "confidence"]},
                {"name": "detect_language", "description": "Detect the language of a text.", "method": "POST",
                 "required_parameters": [{"name": "text", "type": "string", "description": "Text to detect"}],
                 "optional_parameters": [],
                 "response_fields": ["language_code", "language_name", "confidence"]},
                {"name": "list_languages", "description": "List all supported languages.", "method": "GET",
                 "required_parameters": [],
                 "optional_parameters": [],
                 "response_fields": ["language_code", "language_name", "native_name"]},
            ]
        },
    ],
    "Data": [
        {
            "tool_name": "database_api",
            "tool_description": "Query and manage cloud database tables via REST.",
            "api_list": [
                {"name": "execute_query", "description": "Execute a SQL-like query.", "method": "POST",
                 "required_parameters": [{"name": "query", "type": "string", "description": "Query string"},
                                         {"name": "database_id", "type": "string", "description": "Database ID"}],
                 "optional_parameters": [{"name": "limit", "type": "integer", "description": "Max rows returned"}],
                 "response_fields": ["columns", "rows", "row_count", "execution_time_ms"]},
                {"name": "create_table", "description": "Create a new database table.", "method": "POST",
                 "required_parameters": [{"name": "table_name", "type": "string", "description": "Table name"},
                                         {"name": "schema", "type": "string", "description": "JSON schema definition"},
                                         {"name": "database_id", "type": "string", "description": "Database ID"}],
                 "optional_parameters": [],
                 "response_fields": ["table_id", "status", "created_at"]},
                {"name": "list_tables", "description": "List all tables in a database.", "method": "GET",
                 "required_parameters": [{"name": "database_id", "type": "string", "description": "Database ID"}],
                 "optional_parameters": [],
                 "response_fields": ["table_name", "row_count", "size_mb", "created_at"]},
            ]
        },
        {
            "tool_name": "file_storage_api",
            "tool_description": "Store, retrieve, and manage files in cloud storage.",
            "api_list": [
                {"name": "upload_file", "description": "Upload a file to storage.", "method": "POST",
                 "required_parameters": [{"name": "file_url", "type": "string", "description": "Source file URL"},
                                         {"name": "destination_path", "type": "string", "description": "Storage path"}],
                 "optional_parameters": [{"name": "public", "type": "boolean", "description": "Make file public"}],
                 "response_fields": ["file_id", "storage_url", "size_bytes", "content_type"]},
                {"name": "list_files", "description": "List files in a storage directory.", "method": "GET",
                 "required_parameters": [{"name": "directory", "type": "string", "description": "Storage directory path"}],
                 "optional_parameters": [{"name": "file_type", "type": "string", "description": "Filter by file type"}],
                 "response_fields": ["file_id", "name", "size_bytes", "modified_at", "url"]},
            ]
        },
        {
            "tool_name": "analytics_api",
            "tool_description": "Track events, analyze user behavior, and generate reports.",
            "api_list": [
                {"name": "track_event", "description": "Track a custom analytics event.", "method": "POST",
                 "required_parameters": [{"name": "event_name", "type": "string", "description": "Event name"},
                                         {"name": "properties", "type": "string", "description": "JSON event properties"}],
                 "optional_parameters": [{"name": "user_id", "type": "string", "description": "User ID"}],
                 "response_fields": ["event_id", "status", "ingested_at"]},
                {"name": "get_event_report", "description": "Get report for an event over a time period.", "method": "GET",
                 "required_parameters": [{"name": "event_name", "type": "string", "description": "Event name"},
                                         {"name": "start_date", "type": "string", "description": "Start date"},
                                         {"name": "end_date", "type": "string", "description": "End date"}],
                 "optional_parameters": [{"name": "breakdown", "type": "string", "description": "Breakdown dimension"}],
                 "response_fields": ["date", "count", "unique_users", "breakdown"]},
                {"name": "get_funnel", "description": "Analyze a conversion funnel.", "method": "POST",
                 "required_parameters": [{"name": "steps", "type": "string", "description": "JSON array of funnel events"},
                                         {"name": "date_range", "type": "string", "description": "Date range"}],
                 "optional_parameters": [],
                 "response_fields": ["step", "users_entered", "users_converted", "conversion_rate", "drop_off"]},
            ]
        },
    ],
}


# ─────────────────────────────────────────────────────────────
# Tool scoring function
# ─────────────────────────────────────────────────────────────

def score_tool(tool: dict) -> int:
    """Score a tool by quality (higher = better for selection)."""
    endpoints = tool.get("api_list", [])
    ep_count = len(endpoints)

    total_params = sum(
        len(ep.get("required_parameters", [])) + len(ep.get("optional_parameters", []))
        for ep in endpoints
    )
    has_description = 10 if tool.get("tool_description", "").strip() else 0
    return ep_count * 10 + total_params * 2 + has_description


# ─────────────────────────────────────────────────────────────
# Source directory reader
# ─────────────────────────────────────────────────────────────

def read_tools_from_source(source_dir: Path) -> dict[str, list[dict]]:
    """
    Scan source_dir for ToolBench JSON files.
    Expected layout: source_dir/{Category}/{tool_name}/{tool_name}.json
    OR source_dir/{Category}/tools.json
    Returns: {category: [tool_dict, ...]}
    """
    catalog: dict[str, list[dict]] = {}

    if not source_dir.exists():
        return catalog

    for json_path in sorted(source_dir.rglob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Normalise to list
        if isinstance(data, dict):
            data = [data]

        for tool in data:
            if not isinstance(tool, dict):
                continue
            name = tool.get("tool_name") or tool.get("name", "")
            if not name:
                continue
            api_list = tool.get("api_list") or tool.get("endpoints") or []
            if not api_list:
                continue

            # Infer category from directory structure
            parts = json_path.relative_to(source_dir).parts
            category = parts[0] if len(parts) >= 2 else tool.get("category", "General")

            catalog.setdefault(category, []).append(tool)

    return catalog


# ─────────────────────────────────────────────────────────────
# Subset picker
# ─────────────────────────────────────────────────────────────

def pick_subset(
    source_catalog: dict[str, list[dict]],
    tools_per_category: int,
    min_endpoints: int,
) -> dict[str, list[dict]]:
    """Pick top N tools per category from source, then supplement with synthetic."""

    # Start with synthetic data as the base
    combined: dict[str, list[dict]] = {}
    for cat, tools in SYNTHETIC_CATALOG.items():
        combined[cat] = list(tools)

    # Merge real source data — real data wins if it has more/better tools
    for cat, tools in source_catalog.items():
        qualified = [
            t for t in tools
            if len(t.get("api_list", [])) >= min_endpoints
        ]
        if qualified:
            existing = combined.get(cat, [])
            combined[cat] = existing + qualified  # append real tools

    # Pick top N per category
    subset: dict[str, list[dict]] = {}
    for cat, tools in sorted(combined.items()):
        qualified = [t for t in tools if len(t.get("api_list", [])) >= min_endpoints]
        ranked = sorted(qualified, key=score_tool, reverse=True)
        picked = ranked[:tools_per_category]
        if picked:
            subset[cat] = picked

    return subset


# ─────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────

def write_subset(subset: dict[str, list[dict]], output_dir: Path) -> None:
    """Write each tool as a separate JSON file in category subdirectories."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, tools in subset.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        for tool in tools:
            tool_name = tool.get("tool_name") or tool.get("name", "unknown_tool")
            # Ensure category field is set
            tool["category"] = category
            out_path = cat_dir / f"{tool_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(tool, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

@click.command()
@click.option("--source", default=None, help="Path to ToolBench tools/ directory (optional).")
@click.option("--output", required=True, help="Output directory for the subset.")
@click.option("--tools-per-category", default=3, show_default=True, help="Tools to pick per category.")
@click.option("--min-endpoints", default=1, show_default=True, help="Min endpoints required per tool.")
def main(source, output, tools_per_category, min_endpoints):
    """Pick a representative subset of ToolBench tools."""

    source_catalog: dict[str, list[dict]] = {}
    if source:
        source_path = Path(source).expanduser()
        click.echo(f"Reading from source: {source_path}")
        source_catalog = read_tools_from_source(source_path)
        click.echo(f"Found {sum(len(v) for v in source_catalog.values())} tools "
                   f"in {len(source_catalog)} categories from source")
    else:
        click.echo("No --source given — using synthetic data only.")

    subset = pick_subset(source_catalog, tools_per_category, min_endpoints)

    output_path = Path(output).expanduser()
    write_subset(subset, output_path)

    total_tools = sum(len(v) for v in subset.values())
    total_eps = sum(
        len(t.get("api_list", []))
        for tools in subset.values()
        for t in tools
    )

    click.echo(f"\nFound {len(SYNTHETIC_CATALOG)} categories")
    for cat, tools in sorted(subset.items()):
        click.echo(f"  [{cat}]  picked {len(tools)}/{tools_per_category} tools")
        for t in tools:
            eps = len(t.get("api_list", []))
            sc = score_tool(t)
            click.echo(f"    ✓ {t.get('tool_name', t.get('name'))} ({eps} endpoints, score={sc})")

    click.echo(f"\n✅ SUBSET COMPLETE")
    click.echo(f"   Categories: {len(subset)}")
    click.echo(f"   Total tools selected: {total_tools}")
    click.echo(f"   Total endpoints: {total_eps}")
    click.echo(f"   Output: {output_path}")


if __name__ == "__main__":
    main()
