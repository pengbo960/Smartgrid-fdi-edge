# Lightweight Multi-View Edge Framework for FDI Detection

This repository contains the implementation of an MSc dissertation project
for detecting false data injection attacks in smart-grid IoT communications.

## Current status

- [x] Project initialisation
- [x] Normal smart-meter signal generation
- [x] MQTT publishing
- [x] MQTT subscription and CSV collection
- [ ] Attack simulation
- [ ] Multi-view feature extraction
- [ ] Known-attack detection
- [ ] Unknown-attack detection
- [ ] Raspberry Pi deployment

## Run the MQTT testbed

Start the broker:

```bash
mosquitto -v
```

Start the dataset collector:

```bash
python scripts/collect_dataset.py \
  --output data/raw/normal_run.csv
```

Start the simulator:

```bash
python scripts/run_simulator.py \
  --duration 60 \
  --interval 1 \
  --scenario-id normal_01
```

Run the tests:

```bash
pytest -v
```

## Raw dataset schema

Each collected row contains:

- receive and message timestamps
- scenario ID
- device and MQTT client identities
- MQTT topic, QoS, retain flag and payload size
- sequence number
- voltage, current, power and frequency
- attack type and binary attack label