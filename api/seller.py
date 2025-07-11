import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from enum import Enum
import uuid
import time


# Load environment variables from .env file
load_dotenv()

# =============================================================================
# GLOBAL CONFIGURATION - ENHANCED WITH SELLER MANAGEMENT
# =============================================================================

class GlobalConfig:
    """Enhanced global configuration for supply chain optimization with seller management"""

    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required. Please set it in your .env file.")
    
    LLM_MODEL = "gemini-2.0-flash"
    LLM_TEMPERATURE = 0.1

    # Business Configuration
    CURRENT_DATE = datetime(2025, 12, 15)
    CURRENT_MONTH = 12
    LOCATION = "Mumbai, India"
    NUMBER_OF_STORES = 8
    TOTAL_INVENTORY_CAPACITY = 1000

    # Data Configuration
    DATA_DIRECTORY = "supply_chain_data/"
    PRODUCTS_CSV = "products_inventory.csv"
    DAILY_SALES_CSV = "daily_sales_history.csv"
    MONTHLY_SALES_CSV = "monthly_sales_history.csv"
    SELLERS_CSV = "sellers_database.csv"
    SELLER_RESPONSES_CSV = "seller_responses.csv"
    NEGOTIATIONS_CSV = "negotiations_log.csv"

    # Analysis Configuration
    VELOCITY_ANALYSIS_DAYS = [5, 10, 30]
    SEASONALITY_YEARS = 3
    SAFETY_STOCK_FACTOR = 1.5

    # Seller Management Configuration
    MAX_NEGOTIATION_ROUNDS = 5
    SELLER_RESPONSE_TIMEOUT_HOURS = 24
    RESPONSE_WAIT_TIME_SECONDS = 5  # Time to wait for all responses

# Initialize configuration
config = GlobalConfig()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=config.LLM_TEMPERATURE,
    convert_system_message_to_human=True
)

# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================

class ResponseStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    NEGOTIATING = "negotiating"
    COMPLETED = "completed"
    FINAL_ACCEPTED = "final_accepted"

class SellerResponse(BaseModel):
    seller_id: str
    product_id: int
    quantity_requested: int
    response: str  # "yes" or "no"
    expected_delivery_days: Optional[int] = None
    quoted_price: Optional[float] = None
    additional_notes: Optional[str] = ""

class NegotiationMessage(BaseModel):
    seller_id: str
    product_id: int
    message: str
    counter_offer_price: Optional[float] = None
    counter_offer_delivery: Optional[int] = None
    
# =============================================================================
# SELLER DATABASE GENERATION
# =============================================================================

def generate_sellers_database():
    """Generate realistic sellers database for Mumbai, India area"""
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    print("🏪 Generating sellers database...")
    
    # Mumbai area locations
    mumbai_areas = [
        "Andheri East", "Bandra West", "Borivali West", "Dadar East", 
        "Goregaon East", "Kandivali West", "Malad West", "Powai",
        "Thane West", "Vashi Navi Mumbai", "Pune Road", "Kalyan"
    ]
    
    # Product-specific seller types
    seller_data = []
    
    products_info = [
        {"id": 5001, "name": "Mangoes (per kg)", "supplier_types": ["Fruit Wholesaler", "Agricultural Supplier", "Farmers Market"]},
        {"id": 5002, "name": "Air Conditioners (1.5 Ton)", "supplier_types": ["Electronics Distributor", "AC Manufacturer", "Appliance Wholesaler"]},
        {"id": 5003, "name": "Winter Jackets", "supplier_types": ["Clothing Manufacturer", "Textile Distributor", "Garment Supplier"]},
        {"id": 5004, "name": "Sweaters", "supplier_types": ["Knitwear Manufacturer", "Textile Supplier", "Clothing Wholesaler"]},
        {"id": 5005, "name": "Bread (per loaf)", "supplier_types": ["Bakery Supplier", "Food Distributor", "Bread Manufacturer"]},
        {"id": 5006, "name": "Cooking Oil (1L)", "supplier_types": ["Oil Distributor", "Food Wholesaler", "FMCG Supplier"]}
    ]
    
    for product in products_info:
        for i in range(3):  # 3 sellers per product
            seller_id = f"SELL_{product['id']}_{i+1:02d}"
            
            # Generate realistic company details
            supplier_type = product["supplier_types"][i]
            area = np.random.choice(mumbai_areas)
            
            # Generate company names based on type and area
            if "Fruit" in supplier_type or "Agricultural" in supplier_type:
                company_names = [f"Maharashtra {supplier_type}", f"{area} Fresh Supplies", f"Mumbai {supplier_type} Co."]
            elif "Electronics" in supplier_type or "AC" in supplier_type:
                company_names = [f"TechnoMart {area}", f"Cool Air {supplier_type}", f"Mumbai Electronics Hub"]
            elif "Clothing" in supplier_type or "Textile" in supplier_type:
                company_names = [f"Fashion Hub {area}", f"Mumbai {supplier_type}", f"Style Craft Industries"]
            elif "Bakery" in supplier_type or "Bread" in supplier_type:
                company_names = [f"Daily Fresh {area}", f"Mumbai Bakers Association", f"Golden Crust Supplies"]
            else:
                company_names = [f"Supreme {supplier_type}", f"{area} Trading Co.", f"Mumbai {supplier_type}"]
            
            company_name = np.random.choice(company_names)
            
            # Generate reliability and pricing based on position (0=premium, 1=mid, 2=budget)
            if i == 0:  # Premium supplier
                reliability_score = np.random.uniform(85, 95)
                price_factor = np.random.uniform(1.05, 1.15)  # 5-15% higher than base
                delivery_reliability = np.random.uniform(90, 98)
            elif i == 1:  # Mid-range supplier
                reliability_score = np.random.uniform(75, 85)
                price_factor = np.random.uniform(0.95, 1.05)  # Around base price
                delivery_reliability = np.random.uniform(80, 90)
            else:  # Budget supplier
                reliability_score = np.random.uniform(65, 75)
                price_factor = np.random.uniform(0.85, 0.95)  # 5-15% lower than base
                delivery_reliability = np.random.uniform(70, 80)
            
            # Generate contact details
            phone_number = f"+91 {np.random.randint(70000, 99999)}{np.random.randint(10000, 99999)}"
            email = f"{seller_id.lower()}@{company_name.lower().replace(' ', '').replace('.', '')}supplies.com"
            
            seller_data.append({
                "seller_id": seller_id,
                "product_id": product["id"],
                "product_name": product["name"],
                "company_name": company_name,
                "supplier_type": supplier_type,
                "location": area + ", Mumbai",
                "contact_person": f"Manager_{seller_id}",
                "phone_number": phone_number,
                "email": email,
                "reliability_score": round(reliability_score, 1),
                "average_delivery_days": np.random.randint(2, 8) if i == 0 else np.random.randint(3, 10) if i == 1 else np.random.randint(4, 12),
                "price_competitiveness": round(price_factor, 3),
                "delivery_reliability_percent": round(delivery_reliability, 1),
                "payment_terms": np.random.choice(["Net 30", "Net 15", "COD", "Advance 50%"]),
                "minimum_order_quantity": np.random.randint(50, 200) if product["id"] in [5001, 5005, 5006] else np.random.randint(5, 25),
                "created_date": config.CURRENT_DATE.strftime('%Y-%m-%d'),
                "last_updated": config.CURRENT_DATE.strftime('%Y-%m-%d'),
                "status": "active"
            })
    
    sellers_df = pd.DataFrame(seller_data)
    
    # Create directory if it doesn't exist
    os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
    
    # Save to CSV
    sellers_df.to_csv(config.DATA_DIRECTORY + config.SELLERS_CSV, index=False)
    print(f"✅ Created {config.SELLERS_CSV} with {len(sellers_df)} sellers")
    
    return sellers_df

# =============================================================================
# SELLER MANAGEMENT FUNCTIONS
# =============================================================================

def load_sellers_database():
    """Load sellers database from CSV"""
    try:
        df = pd.read_csv(config.DATA_DIRECTORY + config.SELLERS_CSV)
        return df
    except FileNotFoundError:
        print("❌ Sellers CSV not found. Generating database...")
        return generate_sellers_database()

def get_sellers_for_product(product_id: int):
    """Get all sellers for a specific product"""
    sellers_df = load_sellers_database()
    product_sellers = sellers_df[sellers_df['product_id'] == product_id]
    return product_sellers.to_dict('records')

def send_initial_request_to_sellers(product_id: int, quantity_needed: int, product_name: str):
    """Send initial request to all sellers for a product"""
    
    sellers = get_sellers_for_product(product_id)
    
    if not sellers:
        print(f"❌ No sellers found for product {product_id}")
        return []
    
    print(f"📤 Sending requests to {len(sellers)} sellers for {product_name}")
    
    # Create initial request records
    requests_data = []
    
    for seller in sellers:
        request_id = str(uuid.uuid4())
        
        # Generate initial message using LLM
        initial_message = generate_initial_seller_message(
            seller['company_name'],
            product_name,
            quantity_needed,
            seller['contact_person']
        )
        
        request_data = {
            "request_id": request_id,
            "seller_id": seller['seller_id'],
            "product_id": product_id,
            "product_name": product_name,
            "quantity_requested": quantity_needed,
            "initial_message": initial_message,
            "request_date": config.CURRENT_DATE.strftime('%Y-%m-%d %H:%M:%S'),
            "status": ResponseStatus.PENDING,
            "response_deadline": (config.CURRENT_DATE + timedelta(hours=config.SELLER_RESPONSE_TIMEOUT_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        requests_data.append(request_data)
        
        print(f"   📧 Request sent to {seller['company_name']} ({seller['seller_id']})")
        print(f"      Message: {initial_message[:100]}...")
    
    # Save requests to file
    requests_df = pd.DataFrame(requests_data)
    requests_file = config.DATA_DIRECTORY + f"seller_requests_{product_id}_{config.CURRENT_DATE.strftime('%Y%m%d')}.csv"
    requests_df.to_csv(requests_file, index=False)
    
    print(f"💾 Seller requests saved to: {requests_file}")
    
    return requests_data

def generate_initial_seller_message(company_name: str, product_name: str, quantity: int, contact_person: str):
    """Generate initial message to sellers using LLM"""
    
    prompt = f"""
    Generate a concise business inquiry message for a supplier.
    
    DETAILS:
    - Your Company: Mumbai Retail Chain (8 stores)
    - Supplier: {company_name}
    - Contact: {contact_person}
    - Product: {product_name}
    - Quantity: {quantity} units
    - Location: Mumbai, India
    
    Write a brief, professional message (2-3 sentences max) that:
    1. States your requirement clearly
    2. Asks for pricing and delivery time
    3. Mentions potential for ongoing business
    
    Keep it under 80 words, suitable for business communication.
    """
    
    try:
        response = llm.invoke(prompt)
        message = response.content.strip()
        # Clean up any formatting issues
        message = message.replace('[Product Name]', product_name)
        message = message.replace('[Your Company Name]', 'Mumbai Retail Chain')
        message = message.replace('[Supplier Company Name]', company_name)
        message = message.replace('[Supplier Contact Person]', contact_person)
        message = message.replace('[Your Name]', 'Procurement Manager')
        return message
    except Exception as e:
        print(f"Error generating message with LLM: {e}")
        # Fallback message if LLM fails
        return f"Hi {contact_person}, Mumbai Retail Chain needs {quantity} units of {product_name}. Please quote your best price and delivery time. We're looking for reliable suppliers for ongoing business. Thanks!"

# =============================================================================
# FINAL ACCEPTANCE SYSTEM
# =============================================================================

def evaluate_all_negotiations(negotiations: list):
    """Evaluate all completed negotiations and select final winner"""
    
    print("\n📊 Evaluating all completed negotiations...")
    
    # Get product info
    product_id = negotiations[0]['product_id']
    products_df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
    product_info = products_df[products_df['product_id'] == product_id].iloc[0].to_dict()
    
    # Collect final offers from all negotiations
    final_offers = []
    
    for negotiation in negotiations:
        if negotiation['status'] == 'accepted':
            final_offers.append({
                "seller_id": negotiation['seller_id'],
                "company_name": negotiation['company_name'],
                "final_price": negotiation.get('final_price', negotiation['initial_quoted_price']),
                "final_delivery": negotiation.get('final_delivery', negotiation['initial_delivery_days']),
                "negotiation_rounds": negotiation['current_round'],
                "initial_price": negotiation['initial_quoted_price'],
                "price_improvement": negotiation['initial_quoted_price'] - negotiation.get('final_price', negotiation['initial_quoted_price'])
            })
    
    if not final_offers:
        print("❌ No successful negotiations completed")
        return None
    
    # Use LLM to select final winner
    final_winner = select_final_winner_with_llm(final_offers, product_info)
    
    return final_winner

def select_final_winner_with_llm(final_offers: list, product_info: dict):
    """Use LLM to select the final winner from all negotiated offers"""
    
    offers_summary = "\n".join([
        f"- {offer['company_name']}: ₹{offer['final_price']} (was ₹{offer['initial_price']}), "
        f"{offer['final_delivery']} days, negotiated {offer['negotiation_rounds']} rounds"
        for offer in final_offers
    ])
    
    prompt = f"""
    Select the FINAL supplier from these negotiated offers:
    
    BASELINE REQUIREMENTS:
    - Product: {product_info['product_name']}
    - Current Cost: ₹{product_info['cost_price']}
    - Current Lead Time: {product_info['lead_time_days']} days
    
    FINAL NEGOTIATED OFFERS:
    {offers_summary}
    
    Consider:
    1. Final price vs current cost
    2. Delivery time reliability
    3. Negotiation flexibility (willingness to negotiate)
    4. Overall value proposition
    
    Select the best overall supplier for long-term partnership.
    
    Return JSON: {{"selected_company": "<company name>", "reason": "<brief explanation>"}}
    """
    
    try:
        response = llm.invoke(prompt)
        selection = json.loads(response.content.strip())
        
        # Find the selected offer
        for offer in final_offers:
            if selection['selected_company'] in offer['company_name'] or offer['company_name'] in selection['selected_company']:
                print(f"\n🏆 FINAL WINNER: {offer['company_name']}")
                print(f"   Reason: {selection['reason']}")
                offer['selection_reason'] = selection['reason']
                return offer
        
        # Fallback to best price/delivery combination
        return min(final_offers, key=lambda x: x['final_price'] + x['final_delivery'] * 10)
        
    except:
        # Fallback to best price/delivery combination
        return min(final_offers, key=lambda x: x['final_price'] + x['final_delivery'] * 10)

def send_final_acceptance(winner: dict, product_id: int):
    """Send final acceptance message to winning seller"""
    
    print(f"\n📮 Sending final acceptance to {winner['company_name']}...")
    
    # Get product info
    products_df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
    product_info = products_df[products_df['product_id'] == product_id].iloc[0]
    
    # Generate acceptance message using LLM
    acceptance_message = generate_final_acceptance_message(winner, product_info.to_dict())
    
    # Create final acceptance record
    acceptance_record = {
        "acceptance_id": str(uuid.uuid4()),
        "seller_id": winner['seller_id'],
        "company_name": winner['company_name'],
        "product_id": product_id,
        "product_name": product_info['product_name'],
        "final_price": winner['final_price'],
        "final_delivery_days": winner['final_delivery'],
        "quantity": product_info['quantity_in_inventory'],  # Or specific quantity
        "total_value": winner['final_price'] * product_info['quantity_in_inventory'],
        "acceptance_message": acceptance_message,
        "selection_reason": winner.get('selection_reason', 'Best overall value'),
        "acceptance_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "status": ResponseStatus.FINAL_ACCEPTED
    }
    
    # Save acceptance record
    acceptance_file = config.DATA_DIRECTORY + f"final_acceptance_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(acceptance_file, 'w') as f:
        json.dump(acceptance_record, f, indent=2)
    
    print(f"✅ Final acceptance sent and saved to: {acceptance_file}")
    print(f"\n📄 Acceptance Message:")
    print(f"{acceptance_message}")
    
    return acceptance_record

def generate_final_acceptance_message(winner: dict, product_info: dict):
    """Generate professional final acceptance message using LLM"""
    
    prompt = f"""
    Generate a professional final acceptance message for the selected supplier:
    
    WINNER DETAILS:
    - Company: {winner['company_name']}
    - Final Price: ₹{winner['final_price']} per unit
    - Delivery Time: {winner['final_delivery']} days
    - Product: {product_info['product_name']}
    - Quantity: {product_info['quantity_in_inventory']} units
    - Total Value: ₹{winner['final_price'] * product_info['quantity_in_inventory']}
    
    CONTEXT:
    - They reduced price from ₹{winner['initial_price']} to ₹{winner['final_price']}
    - Selection reason: {winner.get('selection_reason', 'Best overall value')}
    
    Write a formal acceptance message that:
    1. Confirms acceptance of their offer
    2. States the agreed terms clearly
    3. Mentions next steps (PO, delivery schedule)
    4. Expresses enthusiasm for partnership
    5. Provides contact for order processing
    
    Keep it professional but warm, under 150 words.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except:
        # Fallback message
        return f"""
Dear {winner['company_name']},

We are pleased to accept your offer for {product_info['quantity_in_inventory']} units of {product_info['product_name']} at ₹{winner['final_price']} per unit with delivery in {winner['final_delivery']} days.

Total Order Value: ₹{winner['final_price'] * product_info['quantity_in_inventory']}

Our procurement team will send the official purchase order within 24 hours. Please confirm your delivery schedule and coordinate with our warehouse team.

We look forward to a successful long-term partnership.

Best regards,
Procurement Team
Mumbai Retail Chain
procurement@mumbairetail.com
"""

def send_rejection_to_other_sellers(negotiations: list, winner: dict, product_id: int):
    """Send polite rejection messages to non-selected sellers"""
    
    print(f"\n📧 Sending notifications to other sellers...")
    
    # Get product info
    products_df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
    product_info = products_df[products_df['product_id'] == product_id].iloc[0]
    
    for negotiation in negotiations:
        if negotiation['seller_id'] != winner['seller_id'] and negotiation['status'] == 'accepted':
            # Generate rejection message
            rejection_message = generate_rejection_message(
                negotiation['company_name'],
                product_info['product_name']
            )
            
            print(f"   📨 Notified {negotiation['company_name']}: {rejection_message[:50]}...")
            
            # Save rejection record
            rejection_record = {
                "seller_id": negotiation['seller_id'],
                "company_name": negotiation['company_name'],
                "message": rejection_message,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # You can save these to a file if needed

def generate_rejection_message(company_name: str, product_name: str):
    """Generate polite rejection message"""
    
    prompt = f"""
    Write a brief, polite message to inform a supplier they were not selected:
    
    - Company: {company_name}
    - Product: {product_name}
    
    Message should:
    1. Thank them for their time and offer
    2. Inform them they weren't selected this time
    3. Express interest in future opportunities
    
    Keep it under 50 words, professional and respectful.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except:
        return f"Thank you {company_name} for your competitive offer on {product_name}. While we've selected another supplier this time, we value your partnership and look forward to future opportunities."

# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def complete_supplier_selection_process(product_id: int, quantity_needed: int):
    """Complete end-to-end supplier selection process"""
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING COMPLETE SUPPLIER SELECTION PROCESS")
    print(f"{'='*60}\n")
    
    # Get product info
    products_df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
    product = products_df[products_df['product_id'] == product_id].iloc[0]
    product_name = product['product_name']
    
    # Step 1: Send requests to all sellers
    print("📤 STEP 1: Sending requests to sellers...")
    sellers = get_sellers_for_product(product_id)
    requests = send_initial_request_to_sellers(product_id, quantity_needed, product_name)
    
    # Step 2: Wait for all responses
    print(f"\n⏳ STEP 2: Waiting for responses from {len(sellers)} sellers...")
    
    # Simulate seller responses (in production, this would be event-driven)
    simulate_seller_responses(product_id, sellers)
    
    # Step 3: Select best sellers using LLM
    print("\n🤖 STEP 3: LLM evaluating all seller responses...")
    best_sellers = select_best_seller_with_llm(product_id)
    
    if not best_sellers:
        print("❌ No suitable sellers found")
        return None
    
    # Get top 3 sellers for negotiation (or all if less than 3)
    sellers_to_negotiate = get_top_sellers_for_negotiation(product_id, 3)
    
    # Step 4: Negotiate with selected sellers
    print(f"\n🤝 STEP 4: Starting negotiations with {len(sellers_to_negotiate)} sellers...")
    negotiations = negotiate_with_all_sellers(product_id, sellers_to_negotiate)
    
    # Simulate negotiation rounds
    completed_negotiations = simulate_negotiation_rounds(negotiations)
    
    # Step 5: Evaluate all negotiations and select winner
    print("\n🏆 STEP 5: Evaluating all negotiations...")
    final_winner = evaluate_all_negotiations(completed_negotiations)
    
    if not final_winner:
        print("❌ No successful negotiations")
        return None
    
    # Step 6: Send final acceptance
    print("\n✉️ STEP 6: Sending final acceptance...")
    acceptance = send_final_acceptance(final_winner, product_id)
    
    # Step 7: Notify other sellers
    send_rejection_to_other_sellers(completed_negotiations, final_winner, product_id)
    
    print(f"\n{'='*60}")
    print(f"✅ SUPPLIER SELECTION PROCESS COMPLETED")
    print(f"{'='*60}\n")
    
    return acceptance

# =============================================================================
# SIMULATION FUNCTIONS FOR TESTING
# =============================================================================

def simulate_seller_responses(product_id: int, sellers: list):
    """Simulate seller responses for testing"""
    
    responses = []
    
    for seller in sellers:
        # Simulate different response scenarios
        if seller['reliability_score'] > 80:
            response = "yes"
            price_variation = np.random.uniform(0.95, 1.05)
            delivery_variation = np.random.randint(-1, 2)
        else:
            response = "yes" if np.random.random() > 0.3 else "no"
            price_variation = np.random.uniform(0.90, 1.10)
            delivery_variation = np.random.randint(-2, 3)
        
        if response == "yes":
            response_data = {
                "response_id": str(uuid.uuid4()),
                "seller_id": seller['seller_id'],
                "product_id": product_id,
                "quantity_requested": 100,  # Example quantity
                "response": response,
                "expected_delivery_days": max(1, seller['average_delivery_days'] + delivery_variation),
                "quoted_price": round(seller['price_competitiveness'] * 100 * price_variation, 2),  # Example base price
                "additional_notes": f"Ready to supply with {seller['payment_terms']} payment terms",
                "response_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "status": ResponseStatus.ACCEPTED
            }
            
            responses.append(response_data)
    
    # Save responses
    if responses:
        responses_df = pd.DataFrame(responses)
        responses_df.to_csv(config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv", index=False)
        print(f"   ✅ Received {len(responses)} positive responses")

def get_top_sellers_for_negotiation(product_id: int, top_n: int = 3):
    """Get top N sellers based on LLM evaluation"""
    
    # This would use the evaluation scores from earlier
    # For now, we'll select based on price and delivery
    responses_df = pd.read_csv(config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv")
    accepted = responses_df[responses_df['response'] == 'yes'].copy()
    
    # Simple scoring for demonstration
    accepted['score'] = (100 / accepted['quoted_price']) + (10 / accepted['expected_delivery_days'])
    top_sellers = accepted.nlargest(min(top_n, len(accepted)), 'score')
    
    return top_sellers.to_dict('records')

def simulate_negotiation_rounds(negotiations: list):
    """Simulate negotiation rounds for testing"""
    
    completed_negotiations = []
    
    for negotiation in negotiations:
        # Load the negotiation
        log_file = config.DATA_DIRECTORY + f"negotiation_{negotiation['negotiation_id']}.json"
        with open(log_file, 'r') as f:
            negotiation_log = json.load(f)
        
        # Simulate 2-3 rounds of negotiation
        rounds = np.random.randint(2, 4)
        
        for round_num in range(rounds):
            if negotiation_log['status'] != 'active':
                break
            
            # Simulate seller response
            current_price = negotiation_log['messages'][-1].get('proposed_price', negotiation_log['initial_quoted_price'])
            current_delivery = negotiation_log['messages'][-1].get('proposed_delivery', negotiation_log['initial_delivery_days'])
            
            # Seller makes counter offer
            seller_response = {
                "message": f"We can offer ₹{current_price * 0.98:.2f} with {current_delivery} days delivery",
                "counter_offer_price": round(current_price * 0.98, 2),
                "counter_offer_delivery": current_delivery
            }
            
            # Process the response
            decision = process_negotiation_round(negotiation_log, seller_response)
            
            if decision['action'] == 'accept':
                print(f"   ✅ Accepted offer from {negotiation_log['company_name']}")
                break
        
        # Reload the updated negotiation
        with open(log_file, 'r') as f:
            completed_negotiations.append(json.load(f))
    
    return completed_negotiations

def save_negotiation_log(negotiation_log: dict):
    """Save negotiation log to file"""
    
    log_file = config.DATA_DIRECTORY + f"negotiation_{negotiation_log['negotiation_id']}.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Convert the entire log
    clean_log = convert_numpy_types(negotiation_log)
    
    with open(log_file, 'w') as f:
        json.dump(clean_log, f, indent=2)

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Multi-Seller Supply Chain Management API",
    description="API for multi-seller portal with negotiation and selection capabilities",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API MODELS
# =============================================================================

class InitiateProcessRequest(BaseModel):
    product_id: int
    quantity_needed: int
    urgency: Optional[str] = "normal"  # normal, urgent, flexible

class SellerResponseSubmit(BaseModel):
    seller_id: str
    product_id: int
    request_id: str
    response: str  # "yes" or "no"
    quoted_price: Optional[float] = None
    expected_delivery_days: Optional[int] = None
    additional_notes: Optional[str] = ""

class NegotiationResponse(BaseModel):
    negotiation_id: str
    seller_id: str
    message: str
    counter_offer_price: Optional[float] = None
    counter_offer_delivery: Optional[int] = None

class ProcessStatus(BaseModel):
    process_id: str
    product_id: int
    status: str
    current_stage: str
    details: dict

# =============================================================================
# SELLER PORTAL ROUTES
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Multi-Seller Supply Chain Management API",
        "version": "2.0.0",
        "endpoints": {
            "seller_portal": "/portal/seller/{seller_id}",
            "buyer_portal": "/portal/buyer",
            "api_docs": "/docs"
        }
    }

@app.get("/portal/seller/{seller_id}")
async def seller_portal(seller_id: str):
    """Serve seller-specific portal"""
    # In production, serve customized HTML based on seller_id
    return FileResponse('seller_portal.html')

@app.get("/portal/buyer")
async def buyer_portal():
    """Serve buyer portal for managing procurement"""
    return FileResponse('buyer_portal.html')

# =============================================================================
# SELLER MANAGEMENT ROUTES
# =============================================================================

@app.get("/api/sellers")
async def get_all_sellers():
    """Get list of all registered sellers"""
    try:
        sellers_df = load_sellers_database()
        sellers = sellers_df.to_dict('records')
        
        # Group by product
        products = {}
        for seller in sellers:
            prod_id = seller['product_id']
            if prod_id not in products:
                products[prod_id] = {
                    "product_id": prod_id,
                    "product_name": seller['product_name'],
                    "sellers": []
                }
            products[prod_id]["sellers"].append(seller)
        
        return {
            "total_sellers": len(sellers),
            "products": list(products.values()),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sellers/{seller_id}")
async def get_seller_details(seller_id: str):
    """Get detailed information about a specific seller"""
    try:
        sellers_df = load_sellers_database()
        seller = sellers_df[sellers_df['seller_id'] == seller_id]
        
        if seller.empty:
            raise HTTPException(status_code=404, detail="Seller not found")
        
        seller_info = seller.iloc[0].to_dict()
        
        # Get active requests for this seller
        active_requests = get_active_requests_for_seller(seller_id)
        
        # Get negotiation history
        negotiation_history = get_seller_negotiation_history(seller_id)
        
        return {
            "seller_info": seller_info,
            "active_requests": active_requests,
            "negotiation_history": negotiation_history,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sellers/{seller_id}/requests")
async def get_seller_requests(seller_id: str):
    """Get all active requests for a seller"""
    try:
        requests = get_active_requests_for_seller(seller_id)
        return {
            "seller_id": seller_id,
            "active_requests": requests,
            "count": len(requests),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# PROCUREMENT PROCESS ROUTES
# =============================================================================

@app.post("/api/procurement/initiate")
async def initiate_procurement_process(request: InitiateProcessRequest):
    """Initiate complete procurement process for a product"""
    try:
        # Generate process ID
        process_id = f"PROC_{request.product_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get product info
        products_df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
        product = products_df[products_df['product_id'] == request.product_id]
        
        if product.empty:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product_name = product.iloc[0]['product_name']
        
        # Start async process (in production, use task queue)
        sellers = get_sellers_for_product(request.product_id)
        requests_sent = send_initial_request_to_sellers(
            request.product_id, 
            request.quantity_needed, 
            product_name
        )
        
        # Create process tracking record
        process_record = {
            "process_id": process_id,
            "product_id": request.product_id,
            "product_name": product_name,
            "quantity_needed": request.quantity_needed,
            "urgency": request.urgency,
            "status": "active",
            "current_stage": "awaiting_responses",
            "sellers_contacted": len(sellers),
            "started_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "expected_completion": (datetime.now() + timedelta(hours=48)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save process record
        save_process_record(process_record)
        
        return {
            "process_id": process_id,
            "message": f"Procurement process initiated for {product_name}",
            "sellers_contacted": len(sellers),
            "expected_completion": process_record["expected_completion"],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/procurement/{process_id}/status")
async def get_procurement_status(process_id: str):
    """Get current status of procurement process"""
    try:
        process_record = load_process_record(process_id)
        
        if not process_record:
            raise HTTPException(status_code=404, detail="Process not found")
        
        # Get detailed status based on current stage
        if process_record["current_stage"] == "awaiting_responses":
            responses_received = count_responses_received(process_record["product_id"])
            details = {
                "responses_received": responses_received,
                "responses_pending": process_record["sellers_contacted"] - responses_received
            }
        elif process_record["current_stage"] == "negotiating":
            active_negotiations = get_active_negotiations_count(process_record["product_id"])
            details = {"active_negotiations": active_negotiations}
        else:
            details = {}
        
        return ProcessStatus(
            process_id=process_id,
            product_id=process_record["product_id"],
            status=process_record["status"],
            current_stage=process_record["current_stage"],
            details=details
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/procurement/{process_id}/advance")
async def advance_procurement_stage(process_id: str):
    """Manually advance procurement to next stage"""
    try:
        process_record = load_process_record(process_id)
        
        if not process_record:
            raise HTTPException(status_code=404, detail="Process not found")
        
        current_stage = process_record["current_stage"]
        
        if current_stage == "awaiting_responses":
            # Move to evaluation stage
            process_record["current_stage"] = "evaluating_responses"
            save_process_record(process_record)
            
            # Trigger evaluation
            best_sellers = select_best_seller_with_llm(process_record["product_id"])
            
            return {
                "message": "Advanced to evaluation stage",
                "sellers_evaluated": len(best_sellers) if isinstance(best_sellers, list) else 1,
                "next_stage": "negotiation"
            }
            
        elif current_stage == "evaluating_responses":
            # Move to negotiation stage
            process_record["current_stage"] = "negotiating"
            save_process_record(process_record)
            
            # Start negotiations
            sellers_to_negotiate = get_top_sellers_for_negotiation(process_record["product_id"], 3)
            negotiations = negotiate_with_all_sellers(process_record["product_id"], sellers_to_negotiate)
            
            return {
                "message": "Advanced to negotiation stage",
                "negotiations_started": len(negotiations),
                "next_stage": "final_selection"
            }
            
        elif current_stage == "negotiating":
            # Move to final selection
            process_record["current_stage"] = "final_selection"
            save_process_record(process_record)
            
            # Trigger final selection
            result = complete_final_selection(process_record["product_id"])
            
            return {
                "message": "Advanced to final selection",
                "winner_selected": result is not None,
                "next_stage": "completed"
            }
        
        else:
            return {"message": "Process already completed or in final stage"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SELLER RESPONSE ROUTES
# =============================================================================

@app.post("/api/responses/submit")
async def submit_seller_response(response: SellerResponseSubmit):
    """Submit seller response to procurement request"""
    try:
        # Validate seller and request
        sellers_df = load_sellers_database()
        seller = sellers_df[sellers_df['seller_id'] == response.seller_id]
        
        if seller.empty:
            raise HTTPException(status_code=404, detail="Seller not found")
        
        # Create response record
        response_data = {
            "response_id": str(uuid.uuid4()),
            "seller_id": response.seller_id,
            "product_id": response.product_id,
            "request_id": response.request_id,
            "response": response.response.lower(),
            "quoted_price": response.quoted_price,
            "expected_delivery_days": response.expected_delivery_days,
            "additional_notes": response.additional_notes,
            "response_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "status": ResponseStatus.ACCEPTED if response.response.lower() == "yes" else ResponseStatus.DECLINED
        }
        
        # Save response
        save_seller_response(response_data)
        
        # Check if all sellers have responded
        sellers_count = len(get_sellers_for_product(response.product_id))
        responses_count = count_responses_received(response.product_id)
        
        all_responded = responses_count >= sellers_count
        
        return {
            "response_id": response_data["response_id"],
            "status": response_data["status"],
            "all_sellers_responded": all_responded,
            "message": "Response submitted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/responses/product/{product_id}")
async def get_product_responses(product_id: int):
    """Get all responses for a product"""
    try:
        responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
        
        if not os.path.exists(responses_file):
            return {"responses": [], "count": 0}
        
        responses_df = pd.read_csv(responses_file)
        responses = responses_df.to_dict('records')
        
        # Add seller info to each response
        sellers_df = load_sellers_database()
        for response in responses:
            seller_info = sellers_df[sellers_df['seller_id'] == response['seller_id']].iloc[0]
            response['company_name'] = seller_info['company_name']
            response['reliability_score'] = seller_info['reliability_score']
        
        return {
            "product_id": product_id,
            "responses": responses,
            "count": len(responses),
            "accepted_count": len([r for r in responses if r['response'] == 'yes'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# NEGOTIATION ROUTES
# =============================================================================

@app.get("/api/negotiations/active")
async def get_active_negotiations():
    """Get all active negotiations"""
    try:
        negotiations = []
        
        # Find all negotiation files
        for file in os.listdir(config.DATA_DIRECTORY):
            if file.startswith('negotiation_') and file.endswith('.json'):
                with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                    neg = json.load(f)
                    if neg['status'] == 'active':
                        negotiations.append({
                            "negotiation_id": neg['negotiation_id'],
                            "seller_id": neg['seller_id'],
                            "company_name": neg['company_name'],
                            "product_id": neg['product_id'],
                            "current_round": neg['current_round'],
                            "last_message": neg['messages'][-1] if neg['messages'] else None
                        })
        
        return {
            "active_negotiations": negotiations,
            "count": len(negotiations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/negotiations/{negotiation_id}")
async def get_negotiation_details(negotiation_id: str):
    """Get detailed negotiation history"""
    try:
        log_file = config.DATA_DIRECTORY + f"negotiation_{negotiation_id}.json"
        
        if not os.path.exists(log_file):
            raise HTTPException(status_code=404, detail="Negotiation not found")
        
        with open(log_file, 'r') as f:
            negotiation = json.load(f)
        
        return negotiation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/negotiations/{negotiation_id}/respond")
async def respond_to_negotiation(negotiation_id: str, response: NegotiationResponse):
    """Submit seller response in negotiation"""
    try:
        # Load negotiation
        log_file = config.DATA_DIRECTORY + f"negotiation_{negotiation_id}.json"
        
        if not os.path.exists(log_file):
            raise HTTPException(status_code=404, detail="Negotiation not found")
        
        with open(log_file, 'r') as f:
            negotiation_log = json.load(f)
        
        # Validate seller
        if response.seller_id != negotiation_log['seller_id']:
            raise HTTPException(status_code=403, detail="Unauthorized seller")
        
        # Process response
        seller_response = {
            "message": response.message,
            "counter_offer_price": response.counter_offer_price,
            "counter_offer_delivery": response.counter_offer_delivery
        }
        
        decision = process_negotiation_round(negotiation_log, seller_response)
        
        return {
            "negotiation_id": negotiation_id,
            "decision": decision['action'],
            "current_round": negotiation_log['current_round'],
            "status": negotiation_log['status'],
            "buyer_response": decision.get('message', '')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ANALYTICS ROUTES
# =============================================================================

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get system-wide analytics"""
    try:
        sellers_df = load_sellers_database()
        
        # Count active processes
        active_processes = 0
        completed_processes = 0
        
        for file in os.listdir(config.DATA_DIRECTORY):
            if file.startswith('process_') and file.endswith('.json'):
                with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                    process = json.load(f)
                    if process['status'] == 'active':
                        active_processes += 1
                    else:
                        completed_processes += 1
        
        # Get negotiations count
        total_negotiations = len([f for f in os.listdir(config.DATA_DIRECTORY) if f.startswith('negotiation_')])
        
        # Calculate average metrics
        analytics = {
            "total_sellers": len(sellers_df),
            "active_sellers": len(sellers_df[sellers_df['status'] == 'active']),
            "total_products": len(sellers_df['product_id'].unique()),
            "active_processes": active_processes,
            "completed_processes": completed_processes,
            "total_negotiations": total_negotiations,
            "sellers_by_product": sellers_df.groupby('product_name').size().to_dict(),
            "average_reliability": sellers_df['reliability_score'].mean(),
            "system_status": "operational",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/seller/{seller_id}")
async def get_seller_analytics(seller_id: str):
    """Get analytics for specific seller"""
    try:
        sellers_df = load_sellers_database()
        seller = sellers_df[sellers_df['seller_id'] == seller_id]
        
        if seller.empty:
            raise HTTPException(status_code=404, detail="Seller not found")
        
        seller_info = seller.iloc[0].to_dict()
        
        # Count responses and negotiations
        total_requests = 0
        accepted_requests = 0
        total_negotiations = 0
        successful_negotiations = 0
        
        # Analyze response files
        for file in os.listdir(config.DATA_DIRECTORY):
            if file.startswith('seller_responses_'):
                df = pd.read_csv(os.path.join(config.DATA_DIRECTORY, file))
                seller_responses = df[df['seller_id'] == seller_id]
                total_requests += len(seller_responses)
                accepted_requests += len(seller_responses[seller_responses['response'] == 'yes'])
        
        # Analyze negotiation files
        for file in os.listdir(config.DATA_DIRECTORY):
            if file.startswith('negotiation_') and file.endswith('.json'):
                with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                    neg = json.load(f)
                    if neg['seller_id'] == seller_id:
                        total_negotiations += 1
                        if neg['status'] == 'accepted':
                            successful_negotiations += 1
        
        analytics = {
            "seller_info": seller_info,
            "performance_metrics": {
                "total_requests_received": total_requests,
                "requests_accepted": accepted_requests,
                "acceptance_rate": (accepted_requests / total_requests * 100) if total_requests > 0 else 0,
                "total_negotiations": total_negotiations,
                "successful_negotiations": successful_negotiations,
                "negotiation_success_rate": (successful_negotiations / total_negotiations * 100) if total_negotiations > 0 else 0
            },
            "last_activity": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# HELPER FUNCTIONS FOR API
# =============================================================================

def get_active_requests_for_seller(seller_id: str):
    """Get all active requests for a seller"""
    active_requests = []
    
    # Check request files
    for file in os.listdir(config.DATA_DIRECTORY):
        if file.startswith('seller_requests_') and file.endswith('.csv'):
            df = pd.read_csv(os.path.join(config.DATA_DIRECTORY, file))
            seller_requests = df[df['seller_id'] == seller_id]
            
            for _, request in seller_requests.iterrows():
                # Check if response exists
                response_exists = check_response_exists(seller_id, request['product_id'])
                
                if not response_exists:
                    active_requests.append({
                        "request_id": request['request_id'],
                        "product_id": request['product_id'],
                        "product_name": request['product_name'],
                        "quantity_requested": request['quantity_requested'],
                        "request_date": request['request_date'],
                        "deadline": request['response_deadline']
                    })
    
    return active_requests

def get_seller_negotiation_history(seller_id: str):
    """Get negotiation history for a seller"""
    negotiations = []
    
    for file in os.listdir(config.DATA_DIRECTORY):
        if file.startswith('negotiation_') and file.endswith('.json'):
            with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                neg = json.load(f)
                if neg['seller_id'] == seller_id:
                    negotiations.append({
                        "negotiation_id": neg['negotiation_id'],
                        "product_id": neg['product_id'],
                        "status": neg['status'],
                        "rounds": neg['current_round'],
                        "final_price": neg.get('final_price'),
                        "created_at": neg['created_at']
                    })
    
    return negotiations

def save_process_record(process_record: dict):
    """Save procurement process record"""
    filename = config.DATA_DIRECTORY + f"process_{process_record['process_id']}.json"
    with open(filename, 'w') as f:
        json.dump(process_record, f, indent=2)

def load_process_record(process_id: str):
    """Load procurement process record"""
    filename = config.DATA_DIRECTORY + f"process_{process_id}.json"
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def count_responses_received(product_id: int):
    """Count responses received for a product"""
    responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
    
    if os.path.exists(responses_file):
        df = pd.read_csv(responses_file)
        return len(df)
    return 0

def get_active_negotiations_count(product_id: int):
    """Count active negotiations for a product"""
    count = 0
    
    for file in os.listdir(config.DATA_DIRECTORY):
        if file.startswith('negotiation_') and file.endswith('.json'):
            with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                neg = json.load(f)
                if neg['product_id'] == product_id and neg['status'] == 'active':
                    count += 1
    
    return count

def check_response_exists(seller_id: str, product_id: int):
    """Check if seller has already responded"""
    responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
    
    if os.path.exists(responses_file):
        df = pd.read_csv(responses_file)
        return len(df[df['seller_id'] == seller_id]) > 0
    return False

def save_seller_response(response_data: dict):
    """Save seller response to file"""
    responses_file = config.DATA_DIRECTORY + f"seller_responses_{response_data['product_id']}.csv"
    response_df = pd.DataFrame([response_data])
    
    if os.path.exists(responses_file):
        response_df.to_csv(responses_file, mode='a', header=False, index=False)
    else:
        response_df.to_csv(responses_file, index=False)

def complete_final_selection(product_id: int):
    """Complete final selection process"""
    # Load all negotiations for this product
    negotiations = []
    
    for file in os.listdir(config.DATA_DIRECTORY):
        if file.startswith('negotiation_') and file.endswith('.json'):
            with open(os.path.join(config.DATA_DIRECTORY, file), 'r') as f:
                neg = json.load(f)
                if neg['product_id'] == product_id:
                    negotiations.append(neg)
    
    if not negotiations:
        return None
    
    # Evaluate and select winner
    final_winner = evaluate_all_negotiations(negotiations)
    
    if final_winner:
        # Send final acceptance
        acceptance = send_final_acceptance(final_winner, product_id)
        
        # Notify other sellers
        send_rejection_to_other_sellers(negotiations, final_winner, product_id)
        
        return acceptance
    
    return None

# =============================================================================
# ENHANCED SELLER SELECTION WITH ALL RESPONSES REQUIRED
# =============================================================================

def wait_for_all_seller_responses(product_id: int, expected_sellers_count: int):
    """Wait for all sellers to respond before processing"""
    
    print(f"⏳ Waiting for all {expected_sellers_count} sellers to respond...")
    
    responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
    
    # Simulate waiting for responses (in real scenario, this would be event-driven)
    max_wait_time = config.RESPONSE_WAIT_TIME_SECONDS
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            responses_df = pd.read_csv(responses_file)
            if len(responses_df) >= expected_sellers_count:
                print(f"✅ All {expected_sellers_count} sellers have responded!")
                return True
        except FileNotFoundError:
            pass
        
        time.sleep(1)
    
    print(f"⏱️ Timeout reached. Proceeding with available responses.")
    return False

def evaluate_seller_with_llm(seller_response: dict, product_info: dict, sellers_df: pd.DataFrame):
    """Use LLM to evaluate seller response comprehensively"""
    
    seller_info = sellers_df[sellers_df['seller_id'] == seller_response['seller_id']].iloc[0]
    
    prompt = f"""
    Evaluate this seller's offer for our supply chain:
    
    CURRENT PRODUCT DETAILS:
    - Product: {product_info['product_name']}
    - Current Cost Price: ₹{product_info['cost_price']}
    - Current Lead Time: {product_info['lead_time_days']} days
    
    SELLER OFFER:
    - Company: {seller_info['company_name']}
    - Quoted Price: ₹{seller_response['quoted_price']}
    - Delivery Time: {seller_response['expected_delivery_days']} days
    - Reliability Score: {seller_info['reliability_score']}%
    - Delivery Reliability: {seller_info['delivery_reliability_percent']}%
    - Additional Notes: {seller_response.get('additional_notes', 'None')}
    
    Provide a comprehensive evaluation score (0-100) considering:
    1. Price competitiveness vs current cost
    2. Delivery time vs current lead time
    3. Seller reliability history
    4. Overall value proposition
    
    Return ONLY a JSON with format: {{"score": <number>, "key_factor": "<main reason>"}}
    """
    
    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip())
        return result['score'], result['key_factor']
    except:
        # Fallback scoring
        price_score = 50 if seller_response['quoted_price'] <= product_info['cost_price'] else 30
        delivery_score = 30 if seller_response['expected_delivery_days'] <= product_info['lead_time_days'] else 20
        reliability_score = seller_info['reliability_score'] * 0.2
        return price_score + delivery_score + reliability_score, "Calculated score"

def select_best_seller_with_llm(product_id: int):
    """Process all seller responses and select the best using LLM"""
    
    # Load seller responses
    responses_file = config.DATA_DIRECTORY + f"seller_responses_{product_id}.csv"
    
    try:
        responses_df = pd.read_csv(responses_file)
    except FileNotFoundError:
        print(f"❌ No responses found for product {product_id}")
        return None
    
    # Filter accepted responses
    accepted_responses = responses_df[responses_df['response'].str.lower() == 'yes']
    
    if len(accepted_responses) == 0:
        print(f"❌ No sellers accepted the request for product {product_id}")
        return None
    
    print(f"📊 Evaluating {len(accepted_responses)} seller responses using LLM...")
    
    # Get product info
    products_df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
    product_info = products_df[products_df['product_id'] == product_id].iloc[0].to_dict()
    
    # Load seller database
    sellers_df = load_sellers_database()
    
    # Evaluate each seller with LLM
    evaluated_sellers = []
    
    for _, response in accepted_responses.iterrows():
        score, key_factor = evaluate_seller_with_llm(response.to_dict(), product_info, sellers_df)
        
        seller_info = sellers_df[sellers_df['seller_id'] == response['seller_id']].iloc[0]
        
        evaluated_seller = {
            "seller_id": response['seller_id'],
            "company_name": seller_info['company_name'],
            "quoted_price": response['quoted_price'],
            "expected_delivery_days": response['expected_delivery_days'],
            "llm_score": score,
            "key_factor": key_factor,
            "reliability_score": seller_info['reliability_score']
        }
        
        evaluated_sellers.append(evaluated_seller)
        print(f"   📈 {seller_info['company_name']}: Score {score} - {key_factor}")
    
    # Let LLM make final selection
    final_selection = make_final_selection_with_llm(evaluated_sellers, product_info)
    
    return final_selection

def make_final_selection_with_llm(evaluated_sellers: list, product_info: dict):
    """Use LLM to make final seller selection"""
    
    sellers_summary = "\n".join([
        f"- {s['company_name']}: ₹{s['quoted_price']}, {s['expected_delivery_days']} days, Score: {s['llm_score']}, {s['key_factor']}"
        for s in evaluated_sellers
    ])
    
    prompt = f"""
    Select the best seller from these evaluated options:
    
    CURRENT BASELINE:
    - Product: {product_info['product_name']}
    - Current Cost: ₹{product_info['cost_price']}
    - Current Lead Time: {product_info['lead_time_days']} days
    
    SELLER OPTIONS:
    {sellers_summary}
    
    Consider overall value, not just price. Select the seller that offers the best combination of:
    - Competitive pricing
    - Reliable delivery
    - Overall business value
    
    Return ONLY the company name of your selected seller.
    """
    
    try:
        response = llm.invoke(prompt)
        selected_company = response.content.strip()
        
        # Find the selected seller
        for seller in evaluated_sellers:
            if seller['company_name'] in selected_company or selected_company in seller['company_name']:
                print(f"🏆 LLM Selected: {seller['company_name']}")
                return seller
        
        # Fallback to highest score
        return max(evaluated_sellers, key=lambda x: x['llm_score'])
    except:
        # Fallback to highest score
        return max(evaluated_sellers, key=lambda x: x['llm_score'])

# =============================================================================
# ENHANCED NEGOTIATION SYSTEM - FULLY LLM DRIVEN
# =============================================================================

def negotiate_with_all_sellers(product_id: int, selected_sellers: list):
    """Start negotiation with all viable sellers using LLM"""
    
    print(f"🤝 Starting negotiations with {len(selected_sellers)} sellers...")
    
    # Get product info for context
    products_df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
    product_info = products_df[products_df['product_id'] == product_id].iloc[0].to_dict()
    
    negotiations = []
    
    for seller in selected_sellers:
        negotiation_id = str(uuid.uuid4())
        
        # LLM decides negotiation strategy
        negotiation_strategy = get_negotiation_strategy(seller, product_info)
        
        # Generate negotiation message
        negotiation_message = generate_negotiation_message_llm(
            seller, product_info, negotiation_strategy
        )
        
        negotiation_log = {
            "negotiation_id": negotiation_id,
            "seller_id": seller['seller_id'],
            "company_name": seller['company_name'],
            "product_id": product_id,
            "initial_quoted_price": seller['quoted_price'],
            "initial_delivery_days": seller['expected_delivery_days'],
            "current_round": 1,
            "status": "active",
            "strategy": negotiation_strategy,
            "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "messages": [
                {
                    "round": 1,
                    "from": "buyer",
                    "message": negotiation_message['message'],
                    "proposed_price": negotiation_message.get('proposed_price'),
                    "proposed_delivery": negotiation_message.get('proposed_delivery'),
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            ]
        }
        
        negotiations.append(negotiation_log)
        
        print(f"   💬 Negotiating with {seller['company_name']}:")
        print(f"      Strategy: {negotiation_strategy['approach']}")
        print(f"      Message: {negotiation_message['message'][:100]}...")
    
    # Save all negotiations
    for negotiation in negotiations:
        save_negotiation_log(negotiation)
    
    return negotiations

def get_negotiation_strategy(seller: dict, product_info: dict):
    """LLM determines negotiation strategy for each seller"""
    
    prompt = f"""
    Determine negotiation strategy for this seller:
    
    PRODUCT BASELINE:
    - Current Cost: ₹{product_info['cost_price']}
    - Current Lead Time: {product_info['lead_time_days']} days
    
    SELLER OFFER:
    - Company: {seller['company_name']}
    - Quoted Price: ₹{seller['quoted_price']} ({((seller['quoted_price'] - product_info['cost_price']) / product_info['cost_price'] * 100):.1f}% difference)
    - Delivery Time: {seller['expected_delivery_days']} days
    - Reliability: {seller['reliability_score']}%
    
    Determine:
    1. Should we negotiate on price, delivery, or both?
    2. What's our target price and delivery time?
    3. How aggressive should we be?
    
    Return JSON: {{"approach": "<price/delivery/both>", "target_price": <number>, "target_delivery": <number>, "aggressiveness": "<low/medium/high>"}}
    """
    
    try:
        response = llm.invoke(prompt)
        strategy = json.loads(response.content.strip())
        return strategy
    except:
        # Fallback strategy
        return {
            "approach": "both",
            "target_price": product_info['cost_price'] * 0.95,
            "target_delivery": product_info['lead_time_days'],
            "aggressiveness": "medium"
        }

def generate_negotiation_message_llm(seller: dict, product_info: dict, strategy: dict):
    """Generate negotiation message based on LLM strategy"""
    
    prompt = f"""
    Generate a negotiation message based on this strategy:
    
    SELLER: {seller['company_name']}
    Current Offer: ₹{seller['quoted_price']}, {seller['expected_delivery_days']} days
    
    STRATEGY:
    - Approach: {strategy['approach']}
    - Target Price: ₹{strategy['target_price']}
    - Target Delivery: {strategy['target_delivery']} days
    - Aggressiveness: {strategy['aggressiveness']}
    
    Write a professional negotiation message that:
    1. Acknowledges their offer
    2. Makes a counter-proposal based on strategy
    3. Emphasizes long-term partnership
    4. Is {strategy['aggressiveness']} in tone
    
    Keep under 100 words. Be specific with numbers.
    """
    
    try:
        response = llm.invoke(prompt)
        message = response.content.strip()
        
        return {
            "message": message,
            "proposed_price": strategy['target_price'] if strategy['approach'] in ['price', 'both'] else None,
            "proposed_delivery": strategy['target_delivery'] if strategy['approach'] in ['delivery', 'both'] else None
        }
    except:
        return {
            "message": f"Thank you for your offer. We're looking for ₹{strategy['target_price']} with {strategy['target_delivery']} days delivery for long-term partnership.",
            "proposed_price": strategy['target_price'],
            "proposed_delivery": strategy['target_delivery']
        }

def process_negotiation_round(negotiation_log: dict, seller_response: dict):
    """Process seller's response and decide next action using LLM"""
    
    # Get product info
    products_df = pd.read_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV)
    product_info = products_df[products_df['product_id'] == negotiation_log['product_id']].iloc[0].to_dict()
    
    # Build negotiation history
    history = "\n".join([
        f"Round {msg['round']} - {msg['from']}: {msg['message']}" 
        for msg in negotiation_log['messages']
    ])
    
    prompt = f"""
    Analyze this negotiation and decide next action:
    
    PRODUCT BASELINE:
    - Current Cost: ₹{product_info['cost_price']}
    - Current Lead Time: {product_info['lead_time_days']} days
    
    NEGOTIATION HISTORY:
    {history}
    
    SELLER'S LATEST RESPONSE:
    Message: {seller_response['message']}
    Counter Price: ₹{seller_response.get('counter_offer_price', 'Not specified')}
    Counter Delivery: {seller_response.get('counter_offer_delivery', 'Not specified')} days
    
    Current Round: {negotiation_log['current_round']} of {config.MAX_NEGOTIATION_ROUNDS}
    
    Decide:
    1. Should we accept this offer?
    2. If not, what counter-offer should we make?
    3. Or should we end negotiations?
    
    Return JSON: {{"action": "<accept/counter/end>", "reason": "<explanation>", "counter_price": <number or null>, "counter_delivery": <number or null>, "message": "<response message>"}}
    """
    
    try:
        response = llm.invoke(prompt)
        decision = json.loads(response.content.strip())
        
        # Add seller response to log
        negotiation_log['messages'].append({
            "round": negotiation_log['current_round'],
            "from": "seller",
            "message": seller_response['message'],
            "proposed_price": seller_response.get('counter_offer_price'),
            "proposed_delivery": seller_response.get('counter_offer_delivery'),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if decision['action'] == 'accept':
            negotiation_log['status'] = 'accepted'
            negotiation_log['final_price'] = seller_response.get('counter_offer_price')
            negotiation_log['final_delivery'] = seller_response.get('counter_offer_delivery')
            
            # Add acceptance message
            negotiation_log['messages'].append({
                "round": negotiation_log['current_round'] + 1,
                "from": "buyer",
                "message": decision['message'],
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "final_deal": True
            })
            
        elif decision['action'] == 'counter' and negotiation_log['current_round'] < config.MAX_NEGOTIATION_ROUNDS:
            negotiation_log['current_round'] += 1
            
            # Add counter message
            negotiation_log['messages'].append({
                "round": negotiation_log['current_round'],
                "from": "buyer",
                "message": decision['message'],
                "proposed_price": decision.get('counter_price'),
                "proposed_delivery": decision.get('counter_delivery'),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        else:
            negotiation_log['status'] = 'ended'
            negotiation_log['messages'].append({
                "round": negotiation_log['current_round'] + 1,
                "from": "buyer",
                "message": decision.get('message', 'Thank you for your time. We will explore other options.'),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "negotiation_ended": True
            })
        
        save_negotiation_log(negotiation_log)
        return decision
        
    except Exception as e:
        print(f"Error in LLM decision: {e}")
        # Fallback decision logic
        if negotiation_log['current_round'] >= config.MAX_NEGOTIATION_ROUNDS:
            return {"action": "end", "reason": "Max rounds reached"}
        else:
            return {
                "action": "counter",
                "reason": "Continuing negotiation",
                "counter_price": product_info['cost_price'] * 0.98,
                "counter_delivery": product_info['lead_time_days'],
                "message": "We appreciate your flexibility. Can we meet at this price point?"
            }

# =============================================================================
# TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🌟 MULTI-SELLER SUPPLY CHAIN MANAGEMENT SYSTEM")
    print("=" * 50)
    
    # Initialize system
    os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
    
    # Generate seller database if not exists
    if not os.path.exists(config.DATA_DIRECTORY + config.SELLERS_CSV):
        generate_sellers_database()
    
    # Create sample product data if not exists
    if not os.path.exists(config.DATA_DIRECTORY + config.PRODUCTS_CSV):
        products_data = {
            'product_id': [5005, 5006],
            'product_name': ['Bread (per loaf)', 'Cooking Oil (1L)'],
            'quantity_in_inventory': [400, 600],
            'current_quantity_in_store': [80, 100],
            'cost_price': [20, 120],
            'selling_price': [35, 150],
            'shelf_life_days': [3, 730],
            'lead_time_days': [7, 7],
            'category': ['NonSeasonal_HighDemand', 'NonSeasonal_RegularDemand']
        }
        pd.DataFrame(products_data).to_csv(config.DATA_DIRECTORY + config.PRODUCTS_CSV, index=False)
    
    print("\n🚀 Starting Multi-Seller Portal API...")
    print("\n📱 Access Points:")
    print("- Seller Portal: http://localhost:8000/portal/seller/{seller_id}")
    print("- Buyer Portal: http://localhost:8000/portal/buyer")
    print("- API Documentation: http://localhost:8000/docs")
    print("\n🛠️ Key API Endpoints:")
    print("\nSELLER ENDPOINTS:")
    print("- GET  /api/sellers - List all sellers")
    print("- GET  /api/sellers/{seller_id} - Seller details")
    print("- GET  /api/sellers/{seller_id}/requests - Active requests")
    print("- POST /api/responses/submit - Submit response")
    print("- POST /api/negotiations/{id}/respond - Negotiation response")
    print("\nBUYER ENDPOINTS:")
    print("- POST /api/procurement/initiate - Start procurement")
    print("- GET  /api/procurement/{id}/status - Process status")
    print("- POST /api/procurement/{id}/advance - Advance stage")
    print("- GET  /api/responses/product/{id} - View responses")
    print("- GET  /api/negotiations/active - Active negotiations")
    print("\nANALYTICS:")
    print("- GET  /api/analytics/overview - System analytics")
    print("- GET  /api/analytics/seller/{id} - Seller analytics")
    print("\n💡 Example Usage:")
    print("1. Buyer initiates procurement: POST /api/procurement/initiate")
    print("2. Sellers view requests: GET /api/sellers/{seller_id}/requests")
    print("3. Sellers submit responses: POST /api/responses/submit")
    print("4. System evaluates and negotiates automatically")
    print("5. View results: GET /api/procurement/{process_id}/status")
    print("\n🌐 Starting server on http://localhost:8000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")