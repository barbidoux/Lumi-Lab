# 🎯 Session Management Guide

## Overview

Lumi provides **focused development sessions** designed for efficient learning and development cycles. Each session is time-boxed with specific objectives to maximize productivity and learning outcomes.

## 🕐 Session Types

### 1. Quick Session (30 minutes)
**Purpose**: Rapid testing and validation  
**Command**: `make session-quick`

```bash
make session-quick
```

**What it does:**
- ✅ Verifies environment setup
- ✅ Creates sample datasets
- ✅ Runs minimal pipeline test
- ✅ Quick evaluation with basic metrics
- ✅ Generates session log

**Perfect for:**
- 🧪 Testing code changes
- ⚡ Validating setup after installation
- 🔧 Quick debugging sessions
- 🎯 Verifying pipeline functionality

**Expected outcomes:**
- Working pipeline confirmation
- Basic performance baseline
- Environment validation

---

### 2. Prototype Session (2 hours)
**Purpose**: Complete tiny model development  
**Command**: `make session-prototype`

```bash
make session-prototype
```

**What it does:**
- ✅ Environment verification
- ✅ Data preparation (quick dataset)
- ✅ Tiny model training (1000 steps)
- ✅ Model evaluation
- ✅ Architecture validation
- ✅ Complete session documentation

**Perfect for:**
- 🚀 Getting first working model
- 🔬 Experimenting with hyperparameters
- 📊 Understanding training dynamics
- 🎓 Learning the complete pipeline

**Expected outcomes:**
- Functional 6M parameter model
- Training curves and metrics
- Basic text generation capability

---

### 3. Experiment Session (4 hours)
**Purpose**: Serious development with small model  
**Command**: `make session-experiment`

```bash
make session-experiment
```

**What it does:**
- ✅ Full data preparation
- ✅ Small model pre-training
- ✅ Supervised fine-tuning
- ✅ Comprehensive evaluation
- ✅ Performance analysis
- ✅ Model quality assessment

**Perfect for:**
- 🎯 Developing production-quality models
- 📈 Comparing different approaches
- 🔬 Advanced experimentation
- 📚 Fine-tuning optimization

**Expected outcomes:**
- High-quality 42M parameter model
- Fine-tuned conversational capabilities
- Detailed performance metrics

---

### 4. Evaluation Session (1 hour)
**Purpose**: Deep analysis of existing models  
**Command**: `make session-evaluation`

```bash
make session-evaluation
```

**What it does:**
- 🔍 Finds latest trained model
- 📊 Comprehensive evaluation
- 🎯 Performance assessment
- 📝 Detailed reporting
- 💡 Improvement recommendations

**Perfect for:**
- 📈 Analyzing model performance
- 🔬 Comparing training runs
- 📊 Understanding model capabilities
- 🎯 Planning next improvements

---

### 5. Debug Session (Interactive)
**Purpose**: Interactive problem solving  
**Command**: `make session-debug`

```bash
make session-debug
```

**What it provides:**
- 🛠️ Environment diagnostics
- 🔧 Pipeline testing tools
- 📊 Resource monitoring
- 🧹 Cleanup utilities
- 💡 Debugging suggestions

**Perfect for:**
- 🚨 Troubleshooting issues
- 🔍 Investigating problems
- ⚡ Quick fixes and tests
- 🧹 System maintenance

---

### 6. Architecture Session (30 minutes)
**Purpose**: Configuration validation  
**Command**: `make session-architecture`

```bash
make session-architecture
```

**What it does:**
- ✅ Validates all configurations
- 🧮 Estimates memory requirements
- ⚖️ Compares with reference architectures
- 📊 Checks LLaMA compliance
- 💡 Provides optimization suggestions

**Perfect for:**
- 🏗️ Designing custom architectures
- ✅ Validating new configurations
- 📊 Understanding model scaling
- 💾 Memory planning

---

## 🎮 Session Workflow

### Starting a Session

1. **Check current status:**
   ```bash
   make session-status
   ```

2. **Choose appropriate session:**
   - First time: `make session-quick`
   - Learning: `make session-prototype`
   - Development: `make session-experiment`
   - Analysis: `make session-evaluation`

3. **Run the session:**
   ```bash
   make session-prototype  # Example
   ```

4. **Review results:**
   - Check session logs in `./sessions/`
   - Review evaluation results
   - Plan next steps

### Session Logs

Every session creates detailed logs in `./sessions/YYYYMMDD_HHMMSS.log`:

```bash
# View latest session log
ls -t ./sessions/*.log | head -1 | xargs cat

# Search for errors in logs
grep -i error ./sessions/*.log

# View session summary
tail -20 ./sessions/*.log
```

## 🛠️ Session Support Commands

### Status and Monitoring

```bash
make session-status        # Check current state
make monitor              # Resource monitoring
make check-env           # Environment verification
```

### Quick Operations

```bash
make evaluate-quick       # Fast evaluation
make assess-performance   # Analyze results
make validate-architecture # Check configs
```

### Cleanup and Maintenance

```bash
make session-cleanup     # Clean session files
make clean              # Clean temp files
make backup             # Save important files
```

## 📋 Session Planning

### Daily Development Schedule

**Morning Session (2h):**
```bash
make session-status      # Check overnight training
make session-evaluation # Analyze results
# Plan improvements based on analysis
```

**Afternoon Session (2h):**
```bash
make session-experiment  # Implement improvements
# Or continue training from checkpoint
```

**Evening Session (30min):**
```bash
make session-quick       # Quick validation
make backup             # Save progress
```

### Weekly Development Cycle

**Monday**: Architecture design and validation
**Tuesday-Wednesday**: Prototype and experiment
**Thursday**: Evaluation and analysis
**Friday**: Optimization and documentation
**Weekend**: Background training of larger models

## 🎯 Session Best Practices

### Before Starting
- ✅ Check GPU availability: `nvidia-smi`
- ✅ Verify disk space: `df -h`
- ✅ Review previous session logs
- ✅ Plan session objectives

### During Session
- 📊 Monitor resources regularly
- 📝 Take notes on observations
- 🔄 Save intermediate results
- ⏰ Respect time boundaries

### After Session
- 📋 Review session logs
- 💾 Backup important results
- 📝 Document learnings
- 🎯 Plan next session

### Environment Management
```bash
# Start fresh environment
make clean && make session-status

# Quick health check
make session-debug

# Resource monitoring
make monitor
```

## 🔧 Customizing Sessions

### Custom Session Times
```bash
SESSION_TIME=60 make session-prototype  # Custom duration
```

### Custom Model Paths
```bash
MODEL_PATH=./my-model make session-evaluation
```

### Custom Configurations
```bash
CONFIG=./config/custom.json make validate-architecture
```

## 🆘 Troubleshooting Sessions

### Common Issues

**Session fails immediately:**
```bash
make check-env           # Verify setup
make clean              # Clear temp files
```

**Out of memory:**
```bash
make session-quick       # Use smaller model
# Or reduce batch sizes in configs
```

**No model found for evaluation:**
```bash
make session-prototype   # Create tiny model first
```

**Slow training:**
```bash
make monitor            # Check resource usage
# Consider using smaller dataset
```

### Emergency Recovery
```bash
make session-cleanup    # Clean problematic files
make backup             # Save what's working
make session-status     # Reassess situation
```

## 📊 Session Metrics

Each session tracks:
- ⏱️ **Duration**: Actual vs expected time
- 🎯 **Success**: Completion of objectives
- 📈 **Progress**: Model quality improvements
- 💾 **Resources**: Memory and disk usage
- 🔄 **Reproducibility**: Deterministic results

---

*Use sessions to maintain focus, track progress, and build models systematically!*