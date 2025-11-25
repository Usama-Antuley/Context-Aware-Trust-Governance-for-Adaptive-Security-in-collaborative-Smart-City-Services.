  # Context-Aware-Trust-Governance-for-Adaptive-Security-in-collaborative-Smart-City-Services.
This repo implements the “Context-Aware Trust Governance for Adaptive Security in Collaborative Smart City Services” framework, featuring real-time environmental sensing, dynamic trust computation, MultiChain-based security, and automated ECC escalation for adaptive smart agriculture security.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![MultiChain](https://img.shields.io/badge/MultiChain-2.3.3-orange)](https://www.multichain.com/)
[![Contiki-NG](https://img.shields.io/badge/Contiki--NG-Latest-green)](https://github.com/contiki-ng/contiki-ng)

## 📋 Complete Table of Contents

- [System Overview](#-system-overview)
- [Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Complete Installation](#-complete-installation-guide)
- [Blockchain Setup](#-multichain-blockchain-setup)
- [Stream Configuration](#-blockchain-streams-configuration)
- [Core Modules](#-core-modules-detailed)
- [Trust Model](#-trust--risk-model-mathematics)
- [Running the System](#-running-the-system)
- [API Documentation](#-api-documentation)
- [Performance Benchmarks](#-performance-benchmarks)
- [Troubleshooting](#-troubleshooting)
- [Monitoring](#-system-monitoring)
- [Citation](#-citation)

---

## 🎯 System Overview

This implementation provides a **complete adaptive security framework** for smart agriculture using:
- **Real-time environmental monitoring** (OpenWeatherMap + SoilGrids APIs)
- **Multi-dimensional trust computation** (Historical + Reputation + Contextual)
- **Dynamic ECC cryptographic tier escalation** (128 → 192 → 256-bit)
- **MultiChain blockchain** for immutable trust records and service registry
- **Smart contract enforcement** of security policies

**Key Innovation**: Automatically escalates cryptographic security when environmental risks increase (e.g., heavy rain >30mm triggers ECC-256 encryption).

---

## 🏗️ System Architecture

```ascii
+---------------------+     +-----------------------+     +-----------------------+
|    Data Layer       |     |   Trust Engine        |     |  Blockchain Layer     |
|                     |     |                       |     |                       |
|  ┌──────────────┐   |     |  ┌─────────────────┐  |     |  ┌─────────────────┐  |
|  │ OpenWeather  │━━━┿━━━━━┿━▶│ Historical     │  |     |  │ Service Registry│  |
|  │   API        │   |     |  │ Trust Engine   │━━┿━━━━━┿━▶│ Stream          │  |
|  └──────────────┘   |     |  └─────────────────┘  |     |  └─────────────────┘  |
|                     |     |                       |     |                       |
|  ┌──────────────┐   |     |  ┌─────────────────┐  |     |  ┌─────────────────┐  |
|  │ SoilGrids    │━━━┿━━━━━┿━▶│ Reputation     │  |     |  │ Trust Records   │  |
|  │   API        │   |     |  │ Trust Engine   │━━┿━━━━━┿━▶│ Stream          │  |
|  └──────────────┘   |     |  └─────────────────┘  |     |  └─────────────────┘  |
|                     |     |                       |     |                       |
|  ┌──────────────┐   |     |  ┌─────────────────┐  |     |  ┌─────────────────┐  |
|  │ IoT Sensors  │━━━┿━━━━━┿━▶│ Contextual     │  |     |  │ Policy Logs     │  |
|  │ (Contiki-NG) │   |     |  │ Trust Engine   │━━┿━━━━━┿━▶│ Stream          │  |
|  └──────────────┘   |     |  └─────────────────┘  |     |  └─────────────────┘  |
+---------------------+     +-----------------------+     +-----------------------+
                                |                                       |
                                v                                       v
+---------------------+     +-----------------------+     +-----------------------+
|  Security Engine    |     |  Crypto Engine        |     |  Policy Engine        |
|                     |     |                       |     |                       |
|  ┌─────────────────┐|     |  ┌─────────────────┐  |     |  ┌─────────────────┐  |
|  │ Risk Assessment │┿━━━━━┿━▶│ ECC Tier       │━━┿━━━━━┿━▶│ Access Control   │  |
|  │   Module        │|     |  │ Selection       │  |     |  │   Engine        │  |
|  └─────────────────┘|     |  │ (128/192/256)   │  |     |  └─────────────────┘  |
|                     |     |  └─────────────────┘  |     |                       |
|  ┌─────────────────┐|     |                       |     |  ┌─────────────────┐  |
|  │ Threat Detection│┿━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━┿━▶│ Session         │  |
|  │   Module        │|     |                       |     |  │ Management      │  |
|  └─────────────────┘|     +-----------------------+     |  └─────────────────┘  |
+---------------------+                                   +-----------------------+
```

---

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04/22.04 LTS (recommended) or Windows 10/11 with WSL2
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **Network**: Internet connection for API calls

### Software Requirements
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git wget curl build-essential libssl-dev

# Windows (WSL2)
wsl --install -d Ubuntu-22.04
```

---

## 🔧 Complete Installation Guide

### Step 1: Clone Repository
```bash
git clone https://github.com/Shahbazdefender/smartcity-trust-framework-2025.git
cd smartcity-trust-framework-2025

# Create virtual environment
python3 -m venv trust_env
source trust_env/bin/activate  # Linux/Mac
# trust_env\Scripts\activate  # Windows

# Install Python dependencies
pip install --upgrade pip
pip install requests numpy pandas matplotlib cryptography
```

### Step 2: API Keys Setup
```bash
# Get free API keys
# 1. OpenWeatherMap: https://openweathermap.org/api
# 2. SoilGrids: No API key required

# Set environment variables
echo 'export OPENWEATHER_API_KEY="your_actual_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Directory Structure Setup
```bash
# Create necessary directories
mkdir -p logs keys blockchain streams

# Verify structure
tree -a
# Expected output:
# .
# ├── main.py
# ├── multichain.py
# ├── updated_Registration.py
# ├── logs/
# ├── keys/
# ├── blockchain/
# └── streams/
```

---

## ⛓️ MultiChain Blockchain Setup

### Step 1: Install MultiChain
```bash
# Download and install MultiChain
cd /tmp
wget https://www.multichain.com/download/multichain-2.3.3.tar.gz
tar -xvzf multichain-2.3.3.tar.gz
cd multichain-2.3.3
sudo mv multichaind multichain-cli /usr/local/bin/

# Verify installation
multichaind --version
# Output: MultiChain 2.3.3 Daemon (community edition)
```

### Step 2: Create Blockchain Networks
We'll create 4 separate blockchains for different services:

```bash
# Create blockchain directories
mkdir -p ~/blockchains/{weather,transport,sf,bank}

# Create and configure each blockchain
for chain in weather transport sf bank; do
    echo "Creating $chain blockchain..."
    multichain-util create $chain
    sed -i 's/default-network-port = [0-9]*/default-network-port = 97'${chain:0:1}'4/' ~/.multichain/$chain/params.dat
    sed -i 's/default-rpc-port = [0-9]*/default-rpc-port = 97'${chain:1:1}'4/' ~/.multichain/$chain/params.dat
done

# Update specific ports
sed -i 's/default-rpc-port = 9744/default-rpc-port = 9724/' ~/.multichain/weather/params.dat
sed -i 's/default-rpc-port = 9744/default-rpc-port = 7202/' ~/.multichain/transport/params.dat
sed -i 's/default-rpc-port = 9744/default-rpc-port = 9582/' ~/.multichain/sf/params.dat
sed -i 's/default-rpc-port = 9744/default-rpc-port = 6458/' ~/.multichain/bank/params.dat
```

### Step 3: Start Blockchain Daemons
```bash
# Start all blockchain daemons
multichaind weather -daemon
multichaind transport -daemon  
multichaind sf -daemon
multichaind bank -daemon

# Check if all are running
multichain-cli weather getinfo
multichain-cli transport getinfo
multichain-cli sf getinfo
multichain-cli bank getinfo
```

### Step 4: Configure RPC Access
```bash
# Generate RPC passwords for each chain
for chain in weather transport sf bank; do
    echo "rpcuser=multichainrpc" > ~/.multichain/$chain/multichain.conf
    echo "rpcpassword=$(openssl rand -base64 32)" >> ~/.multichain/$chain/multichain.conf
    echo "rpcallowip=127.0.0.1" >> ~/.multichain/$chain/multichain.conf
done

# Restart daemons with new configuration
pkill multichaind
sleep 5

multichaind weather -daemon
multichaind transport -daemon
multichaind sf -daemon  
multichaind bank -daemon
```

---

## 📊 Blockchain Streams Configuration

### Step 1: Create Required Streams
Create a Python script to initialize all streams:

```python
# create_streams.py
import multichain
import json

# Blockchain configurations
chains = {
    "weather": {"port": 9724, "password": "2xYe2PpKbiCpuXfHVhJPDUiVuus3k1dinKfMriSCD6dx"},
    "transport": {"port": 7202, "password": "GLKF9X2pi7pLqMcTYE24yGvniKMc9XSHebHPjnbmn5cL"},
    "sf": {"port": 9582, "password": "4WeL79Sh5uwpiT9kGkP4848jNa5jUPoQn31sM3HyYbDv"},
    "bank": {"port": 6458, "password": "45ujj1UkhFS6iVE6ifeo8beW68n9vpk7f48GN4Zb71pb"}
}

# Stream definitions
streams = [
    "Service Registration",
    "TrustStream", 
    "LocalRule",
    "LocalPolicies",
    "LocalServiceAgreement",
    "AuditLog",
    "RiskLog",
    "PerformanceMetrics"
]

for chain_name, config in chains.items():
    print(f"Configuring {chain_name} blockchain...")
    
    try:
        mc = multichain.MultiChainClient("127.0.0.1", config["port"], "multichainrpc", config["password"])
        
        # Create streams
        for stream in streams:
            try:
                result = mc.create('stream', stream, True)
                print(f"  Created stream: {stream}")
            except Exception as e:
                print(f"  Stream {stream} may already exist: {e}")
                
    except Exception as e:
        print(f"Error connecting to {chain_name}: {e}")

print("Stream creation completed!")
```

Run the stream creation script:
```bash
python create_streams.py
```

### Step 2: Verify Stream Creation
```bash
# Check streams on each blockchain
for chain in weather transport sf bank; do
    echo "=== $chain Blockchain Streams ==="
    multichain-cli $chain liststreams
    echo
done
```

---

## 🔧 Core Modules Detailed

### 1. Main Trust Framework (`main.py`)
**Complete adaptive security implementation**

#### Key Components:
```python
# Trust Computation Functions
compute_historical_trust()    # Time-decayed historical performance
compute_reputation_trust()    # Peer feedback weighted by credibility  
compute_contextual_trust()    # Environmental context modifiers
compute_risk()               # Environmental + service risk assessment
compute_overall_trust()      # Weighted combination of all trust components
```

#### Configuration:
```python
# Crop-specific environmental thresholds
CROP_THRESHOLDS = {
    "wheat": {"temp": (15, 35), "humidity": (40, 80), "soil_moisture": (20, 60)},
    "maize": {"temp": (18, 38), "humidity": (50, 85), "soil_moisture": (25, 70)},
    "rice":  {"temp": (20, 40), "humidity": (60, 90), "soil_moisture": (40, 80)},
}

# Context weights for trust computation
CONTEXT_WEIGHTS = {
    "sensor_integrity": 0.25,
    "location_validity": 0.2,
    "environment_alignment": 0.35, 
    "time_freshness": 0.2
}
```

### 2. MultiChain Client (`multichain.py`)
**Robust blockchain interaction library**

#### Key Features:
- SSL/TLS support for secure communication
- Comprehensive error handling
- Stream operations (publish, subscribe, query)
- JSON-RPC protocol implementation

#### Usage Example:
```python
from multichain import MultiChainClient

# Initialize client
mc = MultiChainClient("127.0.0.1", 9724, "multichainrpc", "your_password")

# Publish trust data
txid = mc.publish('TrustStream', 'trust', {'json': trust_data})

# Query data
items = mc.liststreamitems('TrustStream', 10)  # Last 10 items
```

### 3. Service Registration (`updated_Registration.py`)
**Service registry and authentication system**

#### Service Definitions:
```python
services = {
    "CA": {
        "Service name": "CA",
        "Authentication": "123",
        "Authorization": ["weather", "transport", "sf", "bank"],
        "Access Control": ["0"]  # Highest privilege
    },
    "weather": {
        "Service name": "weather", 
        "Authentication": "321",
        "Authorization": ["CA", "transport", "sf", "bank"],
        "Access Control": ["1"]
    }
    # ... more services
}
```

---

## 📐 Trust & Risk Model Mathematics

### 1. Historical Trust (Definition 1)
$$T_{\text{Hist}}(t) = \frac{\sum \delta^{(t - t_j)/\Delta t} \cdot s_{t_j}}{\sum \delta^{(t - t_j)/\Delta t}}$$

**Where:**
- $\delta = 0.9$ (decay factor)
- $\Delta t = 24$ hours (time window)
- $s_{t_j}$ = trust score at time $t_j$

### 2. Reputation Trust (Definition 2) 
$$T_{\text{Rept}}(t) = \frac{\sum c_j \cdot f_j}{\sum c_j}$$

**Where:**
- $c_j$ = credibility of peer $j$
- $f_j$ = feedback from peer $j$

### 3. Contextual Trust (Definition 3)
$$T_{\text{Ctx}}(t) = \min \left(0.7 \times (M_{\text{rain}}^{0.6} \times M_{\text{temp}}^{0.4}), 1\right)$$

**Environmental Modifiers:**
- $M_{\text{rain}} = 1 - \frac{\text{rainfall}}{50}$ (if rain < 50mm else 0)
- $M_{\text{temp}} = 1 - \frac{|T-28|}{10}$ (if |T-28| < 10 else 0)

### 4. Overall Trust & Risk
$$T_{\text{Overall}} = w_1 T_{\text{Hist}} + w_2 T_{\text{Rept}} + w_3 T_{\text{Ctx}}$$
$$R = \frac{R_{\text{Env}} + R_{\text{Service}}}{2}$$

**Dynamic Weights:**
- $w_1 = 0.5 - 0.2R$ (Historical weight decreases with risk)
- $w_2 = 0.3 + 0.1R$ (Reputation weight increases with risk)  
- $w_3 = 1 - w_1 - w_2$ (Contextual weight adapts)

---

## 🚀 Running the System

### Step 1: Service Registration
```bash
# Register all services on blockchain
python updated_Registration.py

# Expected Output:
# Authentication values from last 5 services:
# Service 1 - Name: CA, Authentication: 123
# Service 2 - Name: weather, Authentication: 321
# Service 3 - Name: transport, Authentication: 678
# Service 4 - Name: sf, Authentication: 456  
# Service 5 - Name: bnk, Authentication: 789
```

### Step 2: Real-time Trust Assessment
```bash
# Run continuous trust monitoring
python main.py

# Expected Output:
# [2025] R=0.234 → T_overall=0.812 → ECC-128 → ALLOW
# [2025] R=0.521 → T_overall=0.689 → ECC-192 → ALLOW  
# [2025] R=0.782 → T_overall=0.423 → ECC-256 → DENY
# [2025] R=0.845 → Session TERMINATED → ECC-256 → DENY
```

### Step 3: Performance Benchmarking
```bash
# Run comprehensive performance tests
python main.py --benchmark

# This generates:
# - 11 performance figures (PNG format)
# - CSV datasets for analysis
# - Blockchain performance metrics
```

### Step 4: Continuous Monitoring
```bash
# Create monitoring script
cat > monitor_system.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== System Status $(date) ==="
    python main.py
    sleep 300  # Run every 5 minutes
done
EOF

chmod +x monitor_system.sh
./monitor_system.sh
```

---

## 📊 Performance Benchmarks

### Generated Analysis Files:

| File | Description | Metrics |
|------|-------------|---------|
| **Exp1(a).png** | Fixed ECC tiers throughput | Requests/sec vs Number of requests |
| **Exp1(b).png** | Fixed ECC tiers delay | Delay (ms) vs Number of requests |
| **Exp2(a).png** | Escalation paths throughput | Throughput for 128→192, 128→256, 192→256 |
| **Exp2(b).png** | Escalation paths delay | End-to-end delay for escalation paths |
| **Collab_NoEsc_*.png** | Collaborative execution without escalation | Execution time, throughput, delay |
| **Collab_Esc_*.png** | Collaborative execution with escalation | Performance with tier changes |
| **ECC_Escalation_Timeline.png** | Real-time tier escalation | Risk vs ECC tier over time |

### Expected Performance:

| Scenario | ECC Tier | Throughput | Avg Delay | Use Case |
|----------|----------|------------|-----------|----------|
| Normal conditions | ECC-128 | 18.40 req/s | 1.72 ms | Low risk |
| Moderate risk | ECC-192 | 15.25 req/s | 1.47 ms | Medium risk |
| High risk | ECC-256 | 12.90 req/s | 1.32 ms | Heavy rain, anomalies |
| Escalation 128→256 | Mixed | 10.00 req/s | 2.68 ms | Risk detection |

---

## 🔌 API Documentation

### Trust Framework Methods

#### `compute_historical_trust(service_id, current_time)`
```python
def compute_historical_trust(service_id, now):
    """
    Calculate historical trust using exponential decay.
    
    Args:
        service_id (str): Unique identifier for the service
        now (float): Current timestamp for decay calculation
        
    Returns:
        float: Historical trust score between 0.0 and 1.0
    """
```

#### `select_ecc_tier(risk_score, overall_trust)`
```python
def select_ecc_tier(R, T_overall):
    """
    Dynamically select ECC cryptographic tier based on risk and trust.
    
    Args:
        R (float): Current risk score (0.0 to 1.0)
        T_overall (float): Overall trust score (0.0 to 1.0)
        
    Returns:
        int: ECC tier (128, 192, or 256)
        
    Policy:
        - T_overall < 0.7 OR R > 0.5 → ECC-256
        - R > 0.7 → ECC-256 (high risk)
        - R < 0.3 → ECC-128 (low risk)  
        - Otherwise → ECC-192 (medium risk)
    """
```

### MultiChain Methods

#### `publish(stream_name, key, data)`
```python
def publish(stream, key, data):
    """
    Publish data to blockchain stream.
    
    Args:
        stream (str): Target stream name
        key (str): Key for data retrieval
        data (dict): JSON-serializable data
        
    Returns:
        str: Transaction ID if successful
        
    Raises:
        Exception: If publishing fails
    """
```

#### `liststreamitems(stream_name, count=10, verbose=False)`
```python
def liststreamitems(stream, count=10, verbose=False):
    """
    Retrieve items from blockchain stream.
    
    Args:
        stream (str): Stream name to query
        count (int): Number of items to retrieve
        verbose (bool): Include blockchain metadata
        
    Returns:
        list: Stream items with data and metadata
    """
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. MultiChain Connection Issues
```bash
# Check if daemons are running
ps aux | grep multichaind

# Check RPC configuration
cat ~/.multichain/weather/multichain.conf

# Restart daemons if needed
pkill multichaind
sleep 3
multichaind weather -daemon
multichaind transport -daemon
multichaind sf -daemon
multichaind bank -daemon
```

#### 2. API Key Problems
```python
# Test OpenWeatherMap API
import requests
API_KEY = "your_api_key"
response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q=Islamabad&appid={API_KEY}")
print(response.status_code)  # Should be 200

# Test SoilGrids API  
response = requests.post("https://rest.isric.org/soilgrids/v2.0/properties/query", 
                        json={"lon": 73.0479, "lat": 33.6844, "property": ["wg3"]})
print(response.status_code)  # Should be 200
```

#### 3. Python Package Issues
```bash
# Recreate virtual environment
deactivate
rm -rf trust_env
python3 -m venv trust_env
source trust_env/bin/activate
pip install -r requirements.txt

# Or install individually
pip install requests numpy pandas matplotlib cryptography
```

#### 4. Permission Problems
```bash
# Fix file permissions
sudo chmod -R 755 ~/.multichain/
sudo chown -R $USER:$USER ~/.multichain/

# Check blockchain file permissions
ls -la ~/.multichain/weather/
```

### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
from main.py import compute_historical_trust, compute_risk
test_trust = compute_historical_trust("test_service", time.time())
print(f"Test trust: {test_trust}")
```

---

## 📈 System Monitoring

### Health Check Script
```python
# health_check.py
import multichain
import requests
import json
from datetime import datetime

def system_health_check():
    print(f"=== System Health Check {datetime.now()} ===")
    
    # Check blockchain connections
    chains = {
        "weather": 9724,
        "transport": 7202, 
        "sf": 9582,
        "bank": 6458
    }
    
    for chain, port in chains.items():
        try:
            mc = multichain.MultiChainClient("127.0.0.1", port, "multichainrpc", "password")
            info = mc.getinfo()
            print(f"✓ {chain} blockchain: OK (blocks: {info['blocks']})")
        except:
            print(f"✗ {chain} blockchain: OFFLINE")
    
    # Check APIs
    try:
        response = requests.get("http://api.openweathermap.org/data/2.5/weather?q=London&appid=demo", timeout=10)
        print(f"✓ OpenWeatherMap API: OK (status: {response.status_code})")
    except:
        print("✗ OpenWeatherMap API: OFFLINE")
        
    try:
        response = requests.post("https://rest.isric.org/soilgrids/v2.0/properties/query", timeout=10)
        print(f"✓ SoilGrids API: OK (status: {response.status_code})")
    except:
        print("✗ SoilGrids API: OFFLINE")

if __name__ == "__main__":
    system_health_check()
```

### Performance Monitoring
```bash
# Monitor system resources
watch -n 5 'echo "=== Blockchain Resources ===" && \
ps aux | grep multichaind | grep -v grep && \
echo "=== Memory Usage ===" && \
free -h && echo "=== Disk Usage ===" && df -h'
```

---


## 🆘 Support

### Getting Help
1. **Check logs**: `tail -f logs/trust_log.json`
2. **Verify blockchain**: `multichain-cli weather getinfo`
3. **Test APIs**: Run `health_check.py`
4. **Check issues**: [GitHub Issues Page]
