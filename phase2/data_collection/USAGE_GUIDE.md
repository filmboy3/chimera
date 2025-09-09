# 🎯 Human Squat Data Collection Guide

## Quick Start

### **1. Basic Usage**
```bash
cd phase2/data_collection
python human_squat_recorder.py --participant "your_name" --duration 30
```

### **2. Complete Command Options**
```bash
python human_squat_recorder.py \
  --participant "user001" \
  --duration 30 \
  --output ./human_data
```

---

## 📋 Step-by-Step Process

### **Setup Phase**
1. **Navigate to directory**:
   ```bash
   cd /Users/jonathanschwartz/Documents/Social_Content/ChimeraAscendant/phase2/data_collection
   ```

2. **Position sensors**:
   - 📱 **Phone**: Secure on your thigh (primary sensor)
   - ⌚ **Watch**: Wear on wrist (secondary sensor)
   - 🏃 **Body**: Stand ready to perform squats

3. **Run the recorder**:
   ```bash
   python human_squat_recorder.py --participant "your_name" --duration 30
   ```

### **Recording Phase**
4. **Follow on-screen instructions**:
   ```
   🎯 Human Squat Data Collection
   Participant: your_name
   Duration: 30 seconds
   Output: ./human_data
   
   Instructions:
   1. Position phone on thigh (primary sensor)
   2. Wear watch on wrist (secondary sensor)
   3. Press ENTER to start recording
   4. Perform natural squats for the duration
   5. Recording will stop automatically
   
   Press ENTER to start recording...
   ```

5. **Press ENTER** when ready

6. **Perform squats** for the specified duration:
   - Natural, controlled movements
   - Consistent pace (about 1 squat every 2-3 seconds)
   - Full range of motion
   - Keep phone secure on thigh

7. **Recording completes automatically** after duration

### **Output Phase**
8. **Check results**:
   ```
   ✅ Recording completed!
   📁 Data saved to: ./human_data/human_squat_your_name_20250908_224800.json
   🏃 Ready for next participant
   ```

---

## 📊 What Gets Recorded

### **Sensor Data**
- **Phone IMU**: 6-axis (accelerometer + gyroscope) at 100Hz
- **Watch IMU**: 6-axis (accelerometer + gyroscope) at 100Hz  
- **Barometer**: Pressure readings at 100Hz
- **Timestamps**: Precise timing for all samples

### **Real-Time Analysis**
- **Rep Detection**: Automatic counting during recording
- **Quality Metrics**: Motion consistency and form analysis
- **Rep Timestamps**: Exact timing of detected repetitions

### **Session Metadata**
- Participant ID and session start time
- Total duration and sample count
- Analysis results and quality scores

---

## 📁 Output File Structure

Each recording creates a JSON file with this structure:
```json
{
  "metadata": {
    "participant_id": "your_name",
    "session_start": 1694217280.123,
    "duration": 30.0,
    "sample_rate": 100,
    "total_samples": 3000
  },
  "sensor_data": {
    "phone_imu": [[ax, ay, az, gx, gy, gz], ...],
    "watch_imu": [[ax, ay, az, gx, gy, gz], ...],
    "barometer": [pressure1, pressure2, ...],
    "timestamps": [0.0, 0.01, 0.02, ...]
  },
  "analysis": {
    "rep_count": 12,
    "quality_metrics": {
      "quality_score": 0.85,
      "consistency": 0.82,
      "motion_range": 4.2,
      "avg_intensity": 2.1
    },
    "rep_timestamps": [2.1, 4.3, 6.5, ...]
  }
}
```

---

## 🎯 Best Practices

### **For Accurate Data Collection**
- **Secure phone placement**: Use a thigh band or pocket
- **Consistent positioning**: Same location for all sessions
- **Natural movements**: Don't exaggerate or slow down
- **Full range of motion**: Complete squats, not partial
- **Steady pace**: About 1 rep every 2-3 seconds

### **For Multiple Sessions**
- **Unique participant IDs**: Use descriptive names
- **Consistent duration**: 30 seconds recommended
- **Rest between sessions**: Allow recovery time
- **Document variations**: Note any changes in setup

### **For Data Quality**
- **Check rep counts**: Verify detected vs actual reps
- **Review quality scores**: Aim for >0.7 consistency
- **Monitor file sizes**: Should be ~50-100KB per session
- **Validate timestamps**: Ensure proper timing

---

## 🔧 Troubleshooting

### **Common Issues**
- **"No module found"**: Install dependencies with `pip install numpy matplotlib`
- **Permission errors**: Ensure write access to output directory
- **Low rep detection**: Check phone placement and movement intensity
- **File not created**: Verify output directory exists

### **Data Quality Issues**
- **Low quality scores**: Improve movement consistency
- **Missing reps**: Increase movement intensity or check placement
- **Noisy data**: Ensure phone is securely attached
- **Short duration**: Use at least 20-30 seconds for meaningful data

---

## 📈 Next Steps After Collection

### **1. Validate Data Quality**
```bash
python data_validator.py \
  --synthetic ../../quantumleap-v3/data/synthetic_squat_dataset.h5 \
  --human ./human_data
```

### **2. Train Model on Real Data**
```bash
cd ../training
python transfer_learning.py \
  --pretrained ../../quantumleap-v3/training/trained_model.pth \
  --human_data ../data_collection/human_data
```

### **3. Deploy to iOS**
```bash
cd ../../quantumleap-v3/deployment
python simple_coreml_converter.py \
  --model ../../phase2/training/human_finetuned_model.pth
```

---

## 🎉 Success Metrics

- **Target**: 100+ human sessions collected
- **Quality**: >0.7 average quality score
- **Accuracy**: >90% rep detection accuracy
- **Consistency**: <20% variation in rep timing
- **Coverage**: Multiple participants and conditions

**Ready to collect real human squat data for Phase 2 training!**
