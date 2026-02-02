# Hugging Face Spaces Deployment Checklist

Use this checklist when deploying EXAIM to Hugging Face Spaces.

## Pre-Deployment

### ✅ Files to Upload

- [ ] `app_gradio.py` - Main application file
- [ ] `requirements.txt` - Rename from `requirements-gradio.txt`
- [ ] `README.md` - Rename from `README_GRADIO.md`
- [ ] `exaim_core/` - Entire directory
- [ ] `demos/` - Entire directory (cdss_example and subdirs)
- [ ] `infra/` - Entire directory (llm_registry.py, model_configs.yaml)

### ✅ Files NOT to Upload

- [ ] `.env` - Contains secrets, use HF Spaces secrets instead
- [ ] `.venv/` - Virtual environment (will be created automatically)
- [ ] `__pycache__/` - Python cache files
- [ ] `.git/` - Git repository data
- [ ] `node_modules/` - Node.js dependencies (if any)
- [ ] `demos/frontend/` - Next.js frontend (not needed for Gradio)

## Space Configuration

### ✅ Basic Settings

- [ ] Space name: Choose descriptive name
- [ ] License: MIT (or your preference)
- [ ] SDK: Gradio
- [ ] SDK version: 4.44.0 or latest
- [ ] Hardware: CPU Basic (free tier) or upgrade if needed
- [ ] Visibility: Public or Private

### ✅ Space Secrets (Settings → Repository secrets)

Choose ONE provider and add its API key:

**Option 1: Google Gemini (Recommended)**
- [ ] `GOOGLE_API_KEY` = your_google_api_key
  - Get from: https://makersuite.google.com/app/apikey
  - Cost: ~$0.05 per case
  - Speed: Fast

**Option 2: OpenAI**
- [ ] `OPENAI_API_KEY` = your_openai_api_key
  - Get from: https://platform.openai.com/api-keys
  - Cost: ~$0.15 per case (GPT-4)
  - Speed: Medium

**Option 3: Groq**
- [ ] `GROQ_API_KEY` = your_groq_api_key
  - Get from: https://console.groq.com/keys
  - Cost: Free tier available
  - Speed: Very fast

### ✅ Optional Secrets

- [ ] `DEFAULT_MODEL` - Override default model (optional)
- [ ] `LOG_LEVEL` - Set logging level (optional)

## File Structure Check

Your Space should have this structure:
```
your-space-name/
├── app_gradio.py          ← Main app file
├── requirements.txt       ← Dependencies
├── README.md              ← Space description
├── exaim_core/
│   ├── __init__.py
│   ├── exaim.py
│   ├── buffer_agent/
│   ├── summarizer_agent/
│   ├── token_gate/
│   ├── schema/
│   └── utils/
├── demos/
│   ├── __init__.py
│   └── cdss_example/
│       ├── __init__.py
│       ├── cdss.py
│       ├── agents/
│       ├── graph/
│       └── schema/
└── infra/
    ├── __init__.py
    ├── llm_registry.py
    └── model_configs.yaml
```

## Post-Deployment Testing

### ✅ Initial Tests

- [ ] Space builds successfully (check build logs)
- [ ] App loads in browser
- [ ] No import errors in logs
- [ ] API key detected (check startup logs)

### ✅ Functionality Tests

- [ ] Try Example 1 (cardiac case)
  - [ ] Raw traces appear in left panel
  - [ ] Summaries appear in right panel
  - [ ] Processing completes without errors

- [ ] Try Example 2 (neuro case)
  - [ ] Same checks as above

- [ ] Try Example 3 (peds case)
  - [ ] Same checks as above

- [ ] Try custom case
  - [ ] Enter your own patient case
  - [ ] Verify processing works

### ✅ Performance Checks

- [ ] Processing completes in 30-60 seconds
- [ ] No timeout errors
- [ ] Memory usage acceptable
- [ ] No rate limit errors from API

### ✅ UI/UX Checks

- [ ] Layout looks correct
- [ ] Text is readable
- [ ] Buttons work
- [ ] Examples load correctly
- [ ] Markdown formatting correct

## Troubleshooting

### Build Fails

**Issue**: Requirements installation fails
- [ ] Check `requirements.txt` syntax
- [ ] Remove version conflicts
- [ ] Check package availability

**Issue**: Import errors
- [ ] Verify all directories uploaded
- [ ] Check `__init__.py` files exist
- [ ] Verify file structure matches expected

### Runtime Errors

**Issue**: "API key not found"
- [ ] Verify secret name matches exactly (case-sensitive)
- [ ] Check secret value has no extra spaces
- [ ] Try restarting the Space

**Issue**: "Module not found"
- [ ] Check all source directories uploaded
- [ ] Verify directory structure
- [ ] Look for missing `__init__.py` files

**Issue**: Timeout errors
- [ ] Upgrade Space hardware
- [ ] Use faster model (Groq)
- [ ] Check API rate limits

**Issue**: Out of memory
- [ ] Upgrade to paid tier
- [ ] Reduce concurrent users
- [ ] Optimize model configuration

## Optimization (Optional)

### ✅ Performance Tuning

- [ ] Enable Space caching
- [ ] Upgrade to GPU tier (if using local models)
- [ ] Add request queuing
- [ ] Implement rate limiting

### ✅ User Experience

- [ ] Add loading indicators
- [ ] Improve error messages
- [ ] Add usage statistics
- [ ] Implement user feedback form

### ✅ Monitoring

- [ ] Monitor API costs
- [ ] Track usage patterns
- [ ] Review error logs
- [ ] Collect user feedback

## Launch Checklist

### ✅ Before Public Launch

- [ ] Test thoroughly with all examples
- [ ] Verify no real patient data in examples
- [ ] Add disclaimer about research use only
- [ ] Update README with contact info
- [ ] Add citation information
- [ ] Test on mobile devices
- [ ] Check accessibility

### ✅ After Launch

- [ ] Share link with team
- [ ] Monitor initial usage
- [ ] Watch for errors in logs
- [ ] Respond to user feedback
- [ ] Update documentation as needed

## Maintenance

### ✅ Regular Checks

- [ ] Monitor API costs weekly
- [ ] Check error logs regularly
- [ ] Update dependencies monthly
- [ ] Review and respond to discussions
- [ ] Keep README up to date

### ✅ Updates

- [ ] Test updates in duplicate Space first
- [ ] Document breaking changes
- [ ] Notify users of major updates
- [ ] Maintain backwards compatibility

## Success Criteria

### ✅ Deployment Successful When:

- [ ] Space builds without errors
- [ ] All test cases work
- [ ] Processing time < 90 seconds
- [ ] No critical errors in logs
- [ ] UI renders correctly
- [ ] API costs are acceptable
- [ ] Users can access and use demo

---

## Quick Reference

**Space URL**: https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

**Support**:
- GitHub Issues: [link]
- Email: [your-email]
- HF Discussions: On Space page

**Resources**:
- [Gradio Docs](https://gradio.app/docs/)
- [HF Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Deployment Guide](GRADIO_DEPLOYMENT.md)

---

Last Updated: February 2026
Version: 1.0
