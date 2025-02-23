import os
from dotenv import load_dotenv
import requests
import json
import time
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import aiohttp
import asyncio
from typing import List, Dict
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import openai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.system_program import TransferParams, transfer
from solders.instruction import Instruction as TransactionInstruction, AccountMeta
from solders.transaction import Transaction
from solders.message import MessageV0 as Message
import logging
import random

# Load environment variables
load_dotenv()

# Securely get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SOLSNIFFER_API_KEY = os.getenv("SOLSNIFFER_API_KEY")  # Optional
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")

# Validate required environment variables
required_env_vars = ["OPENAI_API_KEY", "WALLET_PRIVATE_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Configuration
PUMP_FUN_URL = "https://pump.fun/board"
METEORA_URL = "https://meteora.ag/"  # Placeholder
DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/tokens/"
SOLSNIFFER_API = "https://api.solsniffer.com/v1/score/"  # Hypothetical
RAYDIUM_PROGRAM_ID = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")

# Trading Parameters
PRIORITY_FEE_LAMPORTS = 100000  # 0.0001 SOL
BUY_AMOUNT_SOL = 0.1  # 0.1 SOL
SLIPPAGE_PERCENT = 15  # 15%
TAKE_PROFIT_MULTIPLIER = 10  # 10x
MOONBAG_PERCENT = 15  # 15%

# Known ruggers
KNOWN_RUGGERS = {
    "rugger_wallet1": "Known Rugger 1",
    "rugger_wallet2": "Known Rugger 2"
}

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SolanaClient:
    def __init__(self, url="https://api.mainnet-beta.solana.com"):
        self.url = url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    async def get_latest_blockhash(self):
        async with self.session.post(
            self.url,
            json={"jsonrpc": "2.0", "id": 1, "method": "getLatestBlockhash", "params": []}
        ) as response:
            result = await response.json()
            return Pubkey.from_string(result['result']['value']['blockhash'])

    async def send_transaction(self, transaction):
        tx_bytes = bytes(transaction)
        async with self.session.post(
            self.url,
            json={"jsonrpc": "2.0", "id": 1, "method": "sendTransaction", "params": [tx_bytes.hex(), {"encoding": "hex", "preflightCommitment": "confirmed"}]}
        ) as response:
            result = await response.json()
            return result['result']

class SolanaCoinAnalyzer:
    def __init__(self):
        if not WALLET_PRIVATE_KEY:
            raise ValueError("Wallet private key not found in environment variables")
            
        self.solana_client = SolanaClient()
        self.wallet = Keypair.from_base58_string(WALLET_PRIVATE_KEY)
        self.scaler = MinMaxScaler()
        self.coins_data = []
        self.positions = {}
        
        # Selenium setup
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        self.driver = webdriver.Chrome(options=chrome_options)
        
        logger.info("SolanaCoinAnalyzer initialized successfully")

    def __del__(self):
        """Cleanup Selenium driver"""
        try:
            self.driver.quit()
            logger.info("Selenium WebDriver closed successfully")
        except Exception as e:
            logger.error(f"Error quitting WebDriver: {e}")

    async def fetch_pump_fun_data(self) -> List[Dict]:
        """Fetch token data from Pump.fun using Selenium with retries"""
        retries = 3
        for attempt in range(retries):
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.driver.get, PUMP_FUN_URL)
                
                # Wait for page to load
                WebDriverWait(self.driver, 20).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(2, 5))
                
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                tokens = []
                
                # Updated selectors (inspect pump.fun/board for actual classes)
                token_cards = soup.select('.token-item') or soup.select('div[class*="token"]') or soup.select('.card')
                logger.debug(f"Found {len(token_cards)} potential token elements")
                
                for token in token_cards:
                    address_elem = token.find('span', class_='token-address') or token.get('data-address', '')
                    if not address_elem:
                        continue
                    address = address_elem.text.strip() if hasattr(address_elem, 'text') else address_elem
                    
                    market_cap_elem = token.find('span', class_='market-cap') or token.get('data-marketcap', '0')
                    market_cap = market_cap_elem.text if hasattr(market_cap_elem, 'text') else market_cap_elem
                    volume_elem = token.find('span', class_='volume') or token.get('data-volume', '0')
                    volume = volume_elem.text if hasattr(volume_elem, 'text') else volume_elem
                    
                    tokens.append({
                        'address': address,
                        'market_cap': float(str(market_cap).replace('$', '').replace('K', '000').replace('M', '000000') or 0),
                        'volume': float(str(volume).replace('SOL', '').strip() or 0),
                        'created_at': token.get('data-created', ''),
                        'x_handle': token.get('data-x-handle', ''),
                        'creator': token.get('data-creator', ''),
                        'pool_address': token.get('data-pool-address', '')
                    })
                logger.info(f"Fetched {len(tokens)} tokens from Pump.fun on attempt {attempt + 1}")
                if not tokens:
                    logger.debug(f"Page source snippet: {html[:1000]}")
                return tokens
            except Exception as e:
                logger.error(f"Pump.fun fetch error on attempt {attempt + 1}: {e}")
                logger.debug(f"Page source: {self.driver.page_source[:1000]}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    return []

    async def fetch_meteora_data(self) -> List[Dict]:
        """Fetch pool data from Meteora (placeholder)"""
        logger.warning("Meteora API not implemented; returning empty list")
        return []

    async def fetch_dexscreener_data(self, token_addresses: List[str] = None) -> List[Dict]:
        """Fetch token data from Dexscreener"""
        try:
            async with aiohttp.ClientSession() as session:
                if token_addresses and len(token_addresses) == 1:
                    url = f"{DEXSCREENER_API}{token_addresses[0]}"
                else:
                    url = f"{DEXSCREENER_API}So11111111111111111111111111111111111111112"  # SOL default
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Status {response.status}: {await response.text()}")
                    data = await response.json()
                    pairs = []
                    for pair in data.get('pairs', []):
                        if token_addresses and pair['baseToken']['address'] not in token_addresses:
                            continue
                        pairs.append({
                            'address': pair['baseToken']['address'],
                            'price_usd': float(pair.get('priceUsd', 0)),
                            'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                            'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                            'fdv': float(pair.get('fdv', 0)),
                            'pool_address': pair.get('pairAddress', '')
                        })
                    logger.info(f"Fetched {len(pairs)} pairs from Dexscreener")
                    return pairs
        except Exception as e:
            logger.error(f"Dexscreener fetch error: {e}")
            return []

    async def get_solsniffer_score(self, contract_address: str) -> float:
        """Fetch SolSniffer contract score (hypothetical)"""
        if not SOLSNIFFER_API_KEY or SOLSNIFFER_API_KEY.strip() == "":
            logger.info(f"No SolSniffer API key; using default score 90.0 for {contract_address}")
            return 90.0  # Default for testing without API call
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{SOLSNIFFER_API}{contract_address}"
                headers = {"Authorization": f"Bearer {SOLSNIFFER_API_KEY}"}
                async with session.get(url, headers=headers) as response:
                    data = await response.json()
                    score = float(data.get('score', 0))
                    logger.info(f"SolSniffer score for {contract_address}: {score}")
                    return score
        except Exception as e:
            logger.error(f"SolSniffer error for {contract_address}: {e}")
            return 0

    def detect_fake_volume(self, coin: Dict) -> bool:
        """Detect potential fake volume"""
        volume_24h = coin.get('volume_24h', 0)
        liquidity = coin.get('liquidity', 0)
        
        if volume_24h > 0 and liquidity == 0:
            logger.warning(f"Fake volume detected for {coin.get('address', 'unknown')}: Volume with no liquidity")
            return True
        if liquidity > 0 and volume_24h > liquidity * 10:
            logger.warning(f"Fake volume detected for {coin.get('address', 'unknown')}: High volume ratio")
            return True
        return False

    async def collect_all_data(self) -> List[Dict]:
        """Collect and merge data from all sources"""
        pump_fun_data = await self.fetch_pump_fun_data()
        meteora_data = await self.fetch_meteora_data()
        
        token_addresses = list(set(
            [coin['address'] for coin in pump_fun_data + meteora_data if coin.get('address')]
        ))
        
        dexscreener_data = await self.fetch_dexscreener_data(token_addresses or None)
        
        if not (pump_fun_data or meteora_data or dexscreener_data):
            logger.warning("All data sources failed; returning empty list")
            return []
        
        merged_data = {}
        for data in (pump_fun_data, meteora_data, dexscreener_data):
            for item in data:
                address = item.get('address')
                if address:
                    if address not in merged_data:
                        merged_data[address] = {}
                    merged_data[address].update(item)
        
        filtered_data = []
        for address, coin_data in merged_data.items():
            if not self.detect_fake_volume(coin_data):
                score = await self.get_solsniffer_score(address)
                coin_data['solsniffer_score'] = score
                filtered_data.append(coin_data)
        
        self.coins_data = filtered_data
        return filtered_data

    def preprocess_data(self, data: List[Dict]) -> pd.DataFrame:
        """Preprocess collected data"""
        df = pd.DataFrame(data)
        numeric_cols = ['market_cap', 'volume', 'liquidity', 'price_usd', 
                       'volume_24h', 'fdv', 'solsniffer_score']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if set(numeric_cols).issubset(df.columns):
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df

    def calculate_coin_score(self, coin: Dict) -> float:
        """Calculate a coin's potential score"""
        try:
            weights = {
                'solsniffer_score': 0.4,
                'liquidity': 0.2,
                'volume_24h': 0.2,
                'market_cap': 0.1,
                'price_usd': 0.1
            }
            
            score = 0
            for metric, weight in weights.items():
                if metric in coin:
                    score += coin[metric] * weight
            
            return min(max(score, 0), 1)
        except Exception as e:
            logger.error(f"Error calculating score: {e}")
            return 0

    async def buy_token(self, token_address: str, pool_address: str, price_usd: float) -> bool:
        """Execute buy transaction on Raydium (simplified)"""
        try:
            lamports = int(BUY_AMOUNT_SOL * 1_000_000_000)
            min_output = int(lamports / price_usd * (1 - SLIPPAGE_PERCENT / 100))
            
            # Use AccountMeta for accounts
            swap_instruction = TransactionInstruction(
                program_id=RAYDIUM_PROGRAM_ID,
                accounts=[
                    AccountMeta(pubkey=self.wallet.pubkey(), is_signer=True, is_writable=True),
                    AccountMeta(pubkey=Pubkey.from_string(pool_address), is_signer=False, is_writable=True),
                    AccountMeta(pubkey=Pubkey.from_string(token_address), is_signer=False, is_writable=True)
                ],
                data=bytes([2]) + lamports.to_bytes(8, 'little') + min_output.to_bytes(8, 'little')
            )
            
            # Updated to use MessageV0.compile
            message = Message.compile(
                instructions=[swap_instruction],
                payer=self.wallet.pubkey(),
                recent_blockhash=await self.solana_client.get_latest_blockhash()
            )
            
            tx = Transaction.new_signed_with_payer(
                instructions=[swap_instruction],
                payer=self.wallet.pubkey(),
                signers=[self.wallet],
                recent_blockhash=await self.solana_client.get_latest_blockhash()
            )
            tx_response = await self.solana_client.send_transaction(tx)
            logger.info(f"Buy tx for {token_address}: {tx_response}")
            
            self.positions[token_address] = {
                'amount': min_output / 1_000_000_000,
                'buy_price': price_usd,
                'pool_address': pool_address
            }
            return True
        except Exception as e:
            logger.error(f"Buy error for {token_address}: {e}")
            return False

    async def sell_token(self, token_address: str, pool_address: str, current_price: float) -> bool:
        """Execute sell transaction on Raydium (simplified)"""
        try:
            if token_address not in self.positions:
                logger.warning(f"No position to sell for {token_address}")
                return False
            
            position = self.positions[token_address]
            sell_amount = position['amount'] * (1 - MOONBAG_PERCENT / 100)
            lamports_out = int(sell_amount * current_price * 1_000_000_000 * (1 - SLIPPAGE_PERCENT / 100))
            
            # Use AccountMeta for accounts
            sell_instruction = TransactionInstruction(
                program_id=RAYDIUM_PROGRAM_ID,
                accounts=[
                    AccountMeta(pubkey=self.wallet.pubkey(), is_signer=True, is_writable=True),
                    AccountMeta(pubkey=Pubkey.from_string(pool_address), is_signer=False, is_writable=True),
                    AccountMeta(pubkey=Pubkey.from_string(token_address), is_signer=False, is_writable=True)
                ],
                data=bytes([2]) + int(sell_amount * 1_000_000_000).to_bytes(8, 'little') + lamports_out.to_bytes(8, 'little')
            )
            
            # Updated to use MessageV0.compile
            message = Message.compile(
                instructions=[sell_instruction],
                payer=self.wallet.pubkey(),
                recent_blockhash=await self.solana_client.get_latest_blockhash()
            )
            
            tx = Transaction.new_signed_with_payer(
                instructions=[sell_instruction],
                payer=self.wallet.pubkey(),
                signers=[self.wallet],
                recent_blockhash=await self.solana_client.get_latest_blockhash()
            )
            tx_response = await self.solana_client.send_transaction(tx)
            logger.info(f"Sell tx for {token_address}: {tx_response}")
            
            self.positions[token_address]['amount'] *= (MOONBAG_PERCENT / 100)
            return True
        except Exception as e:
            logger.error(f"Sell error for {token_address}: {e}")
            return False

    async def analyze_coins(self):
        """Main analysis and trading loop"""
        try:
            logger.info(f"Starting analysis at {datetime.now()}")
            
            data = await self.collect_all_data()
            if not data:
                logger.info("No valid coins found after filtering")
                return
            
            df = self.preprocess_data(data)
            
            for _, coin in df.iterrows():
                try:
                    if coin['address'] in self.positions:
                        current_price = coin['price_usd']
                        position = self.positions[coin['address']]
                        
                        if current_price >= position['buy_price'] * TAKE_PROFIT_MULTIPLIER:
                            logger.info(f"Take profit triggered for {coin['address']}")
                            await self.sell_token(
                                coin['address'], 
                                position['pool_address'], 
                                current_price
                            )
                        continue
                    
                    score = self.calculate_coin_score(coin)
                    logger.info(f"Coin {coin['address']} score: {score}")
                    
                    if score > 0.8:
                        logger.info(f"High potential coin found: {coin['address']}")
                        await self.buy_token(
                            coin['address'],
                            coin['pool_address'],
                            coin['price_usd']
                        )
                
                except Exception as e:
                    logger.error(f"Error analyzing coin {coin.get('address', 'unknown')}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in analyze_coins: {e}")

async def main():
    try:
        analyzer = SolanaCoinAnalyzer()
        async with analyzer.solana_client:
            while True:
                try:
                    await analyzer.analyze_coins()
                    logger.info("Waiting 5 minutes for next scan...")
                    await asyncio.sleep(300)
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    await asyncio.sleep(60)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")