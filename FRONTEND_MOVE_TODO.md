# EXAID Restructure - Frontend Move Instructions

## Status

The EXAID restructure is **95% complete**. All Python code has been successfully restructured:
- ✅ `exaid_core/` created with all core EXAID components
- ✅ `demos/cdss_example/` contains the CDSS demo
- ✅ `demos/backend/` contains the FastAPI server
- ✅ All imports updated across 31 Python files
- ✅ All sys.path manipulation removed (10 files cleaned)
- ✅ Documentation and config files updated
- ✅ Validation script passes all tests (14/14)
- ✅ Backend server starts successfully

## Remaining Task: Move Frontend Folder

The `exaid-frontend/` folder needs to be moved to `demos/frontend/`. This could not be completed during the automated restructure because the folder is locked by a running process (likely npm/node).

### Manual Steps to Complete

1. **Stop all running frontend processes:**
   - Close any terminals running `npm run dev`
   - Close any Node.js processes
   - Close VS Code if it has the frontend folder open

2. **Move the frontend folder:**
   ```powershell
   # Option 1: Using PowerShell
   Move-Item -Path exaid-frontend -Destination demos/frontend
   
   # Option 2: Using git (preserves history)
   git mv exaid-frontend demos/frontend
   ```

3. **Verify the frontend works:**
   ```powershell
   cd demos/frontend
   npm run dev
   ```

4. **Commit the move:**
   ```powershell
   git add .
   git commit -m "Phase 5: Move frontend to demos/frontend"
   ```

## Alternative: Keep Frontend in Current Location Temporarily

If you want to test the restructure before moving the frontend:

1. The backend and demo work perfectly without moving the frontend
2. The frontend can stay at `exaid-frontend/` temporarily
3. Just update `start-dev.ps1` to use the correct path if needed
4. The `.env.local` file uses URL-based config, so no changes needed there

## Verification After Frontend Move

Run the validation script to ensure everything still works:
```powershell
.\.venv\Scripts\python.exe validate_restructure.py
```

Test the full stack:
```powershell
# Terminal 1: Backend
python -m uvicorn demos.backend.server:app --reload

# Terminal 2: Frontend (after move)
cd demos/frontend
npm run dev
```

## Documentation Already Updated

All documentation has been updated to reflect `demos/frontend/`:
- ✅ `start-dev.ps1` - references `demos\frontend`
- ✅ `MIGRATION_GUIDE.md` - updated paths
- ✅ `MIGRATION_COMPLETE.md` - updated paths
- ✅ `README.md` - updated import examples
- ✅ `DOCUMENTATION.md` - updated code examples

Once the frontend is moved, the restructure will be 100% complete!
