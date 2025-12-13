# Deployment Checklist

Use this checklist before deploying or demonstrating the web application.

## Pre-Deployment Checks

### Infrastructure
- [ ] Hadoop services running (`jps` shows NameNode, DataNode)
- [ ] HDFS accessible (`hdfs dfs -ls /`)
- [ ] HDFS temporary directory created
- [ ] Sufficient disk space (5GB+ free)
- [ ] Sufficient RAM (4GB+ free for application)

### Backend
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Model file exists and is correct size (~94MB)
- [ ] `config.py` paths configured correctly
- [ ] `tumor_descriptions.json` exists in `shared/` folder
- [ ] Backend starts without errors
- [ ] Health endpoint responds (`curl http://localhost:5000/api/health`)

### Frontend
- [ ] Node modules installed (`npm install`)
- [ ] Frontend starts without errors
- [ ] Can access at `http://localhost:3000`
- [ ] Backend status shows "Online" (green indicator)

## Functional Testing

### Upload Functionality
- [ ] Drag-and-drop works
- [ ] Click-to-upload works
- [ ] Image preview displays correctly
- [ ] Invalid file types rejected (e.g., .txt, .pdf)
- [ ] Large files rejected (>10MB)

### Classification
- [ ] Classification completes in < 15 seconds
- [ ] Progress indicator shows during processing
- [ ] Results display correctly
- [ ] All 4 tumor types can be classified
- [ ] Confidence scores are reasonable (not all 25%)

### Visualization
- [ ] Original image displays
- [ ] Heatmap overlay displays
- [ ] Toggle between views works
- [ ] Heatmap shows reasonable attention (not uniform)
- [ ] Images are properly sized/scaled

### Analysis
- [ ] Classification result shown
- [ ] Confidence percentage displayed
- [ ] Probability breakdown shown
- [ ] Medical description appears
- [ ] Model interpretation text present
- [ ] Disclaimer visible

### Error Handling
- [ ] Invalid file type shows error message
- [ ] Backend offline shows warning
- [ ] Network errors handled gracefully
- [ ] Error messages are user-friendly

## Performance Testing

- [ ] First classification: < 15 seconds
- [ ] Subsequent classifications: < 10 seconds
- [ ] No memory leaks (check with multiple classifications)
- [ ] Browser doesn't freeze during processing
- [ ] Smooth animations and transitions

## Browser Compatibility

Test in multiple browsers:
- [ ] Chrome/Chromium
- [ ] Firefox
- [ ] Safari (if on Mac)
- [ ] Edge

## Responsive Design

Test on different screen sizes:
- [ ] Desktop (1920x1080)
- [ ] Laptop (1366x768)
- [ ] Tablet (768x1024)
- [ ] Mobile (375x667)

## Security Checks

- [ ] File validation working
- [ ] No console errors exposing sensitive info
- [ ] CORS configured correctly
- [ ] No hardcoded credentials in code

## Documentation

- [ ] README.md complete and accurate
- [ ] Setup instructions tested
- [ ] Troubleshooting section helpful
- [ ] API documentation matches implementation
- [ ] Code comments are clear

## Demo Preparation

- [ ] Test images ready (one from each class)
- [ ] Backend and frontend running
- [ ] Browser tabs organized
- [ ] Presentation points prepared
- [ ] Backup plan if demo fails

## Post-Deployment

- [ ] Monitor logs for errors
- [ ] Check HDFS storage not filling up
- [ ] Verify temporary files being cleaned up
- [ ] Note any performance issues
- [ ] Collect user feedback

---

## Emergency Procedures

### Backend Crash
1. Check Flask terminal for error message
2. Restart Flask: `python app.py`
3. Verify model loaded correctly
4. Check Hadoop services still running

### Frontend Crash
1. Check browser console (F12)
2. Refresh page
3. Clear browser cache if needed
4. Restart React: `npm start`

### Hadoop Issues
1. Check services: `jps`
2. Restart if needed: `stop-dfs.sh && start-dfs.sh`
3. Verify HDFS: `hdfs dfs -ls /`
4. Check logs in `$HADOOP_HOME/logs/`

### Out of Memory
1. Close other applications
2. Reduce Spark memory in `config.py`
3. Restart backend
4. Monitor with `top` or Task Manager

---

## Quick Commands Reference

```bash
# Check Hadoop
jps

# Start Hadoop
start-dfs.sh

# Start Backend
cd webapp/backend && source venv/bin/activate && python app.py

# Start Frontend
cd webapp/frontend && npm start

# Test Health
curl http://localhost:5000/api/health

# Run Integration Tests
cd webapp && python test_integration.py

# Check Ports
lsof -i :5000  # Backend
lsof -i :3000  # Frontend
```

---

**All checks passed? You're ready to deploy!**
