#!/bin/bash
# Training Progress Monitor
# Usage: ./monitor_training.sh

echo "=================================================================================="
echo "TRAINING PROGRESS MONITOR"
echo "=================================================================================="
echo ""

# Check if training is running
if ps aux | grep -E "python.*train.py" | grep -v grep > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
    
    # Show process info
    echo "Process Information:"
    ps aux | grep -E "python.*train.py" | grep -v grep | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Memory:", $4"%", "| Runtime:", $10}'
    echo ""
else
    echo "⏸️  Training is NOT running"
    echo ""
fi

# Check training log
LOG_FILE="data/processed/logs/training.log"
if [ -f "$LOG_FILE" ]; then
    echo "Latest Training Progress:"
    echo "--------------------------------------------------------------------------------"
    
    # Show latest epoch
    LATEST_EPOCH=$(tail -100 "$LOG_FILE" | grep "Epoch.*Train Loss" | tail -1)
    if [ ! -z "$LATEST_EPOCH" ]; then
        echo "Latest Epoch:"
        echo "  $LATEST_EPOCH"
        echo ""
        
        # Extract and show progress
        EPOCH_NUM=$(echo "$LATEST_EPOCH" | grep -oP 'Epoch \K\d+' | head -1)
        TOTAL_EPOCHS=$(echo "$LATEST_EPOCH" | grep -oP '/\K\d+' | head -1)
        if [ ! -z "$EPOCH_NUM" ] && [ ! -z "$TOTAL_EPOCHS" ]; then
            PROGRESS=$(echo "scale=1; $EPOCH_NUM * 100 / $TOTAL_EPOCHS" | bc)
            echo "Progress: $EPOCH_NUM/$TOTAL_EPOCHS epochs ($PROGRESS%)"
            echo ""
        fi
        
        echo "Recent Epochs (last 5):"
        tail -100 "$LOG_FILE" | grep "Epoch.*Train Loss" | tail -5 | while read line; do
            echo "  $line"
        done
    else
        echo "Training is in data loading phase..."
        echo ""
        echo "Recent activity:"
        tail -20 "$LOG_FILE" | grep -E "Processed|Total windows|Step" | tail -5
    fi
    echo ""
else
    echo "Training log not found yet."
    echo ""
fi

# Check if training completed
if grep -q "TRAINING COMPLETED" "$LOG_FILE" 2>/dev/null; then
    echo "=================================================================================="
    echo "✅ TRAINING COMPLETED!"
    echo "=================================================================================="
    echo ""
    echo "Final Results:"
    grep -A 5 "TRAINING COMPLETED" "$LOG_FILE" | head -6
    echo ""
fi

echo "=================================================================================="
echo "MONITORING COMMANDS:"
echo "=================================================================================="
echo "Live monitoring (epochs only):"
echo "  tail -f $LOG_FILE | grep -E 'Epoch|Train Loss|Val Loss|Saved best'"
echo ""
echo "Live monitoring (full log):"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check training status:"
echo "  ps aux | grep train.py | grep -v grep"
echo ""
echo "View this monitor:"
echo "  ./monitor_training.sh"
echo "=================================================================================="

