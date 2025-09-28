// static/js/vj_playground.js
// Image jigsaw with slot-based palette ordering (no reflow).
// Video / 3D keep previous behavior. If you want slot-based for them too, tell me.

(function(){
    const S = {};
    const $  = (sel, root=document) => root.querySelector(sel);
    const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
  
    // --------- State ---------
    S.mode = 'image';          // 'image' | 'video' | 'view3d'
    S.gridN = 3;
  
    // Image
    S.imageSrc = null;
    S.answerOrder = [];        // correct ids 0..N*N-1
    S.originalTiles = [];      // [{id, src}]
  
    // Video
    S.videoClips = [];         // [{id, src}]
    S.videoAnswer = [0,1,2,3,4,5];
  
    // 3D
    S.threeDImage = null;
    S.depthOrderCorrect = [1,2,3,4,5,6];
    S.depthTokens = [1,2,3,4,5,6];
  
    // --------- Refs ---------
    function refs(){
      return {
        // meta
        score: $('#vj-score'),
        hint:  $('#vj-hint'),
  
        // controls
        gridSizeSel: $('#vj-grid-size'),
        btnShuffle:  $('#vj-shuffle'),
        btnReset:    $('#vj-reset'),
        btnCheck:    $('#vj-check'),
  
        // tabs/panels
        tabs:   $('#vj-tabs'),
        panels: $('#vj-panels'),
  
        // image DOM
        paletteImg:  $('#vj-palette'),
        gridImg:     $('#vj-grid'),
  
        // video DOM (unchanged)
        paletteVideo: $('#vj-video-palette'),
        timeline:     $('#vj-video-timeline'),
  
        // 3D DOM (unchanged)
        threeDImage:  $('#vj-3d-image'),
        palette3D:    $('#vj-3d-palette'),
        line3D:       $('#vj-3d-line'),
      };
    }
  
    // --------- Utils ---------
    function setCols(el, n){ if(el) el.style.gridTemplateColumns = `repeat(${n}, 1fr)`; }
    function setAR(el, ar){ if(el) el.style.setProperty('--ar', String(ar)); } // ar = height/width
    function shuffle(arr){ for(let i=arr.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [arr[i],arr[j]]=[arr[j],arr[i]]; } return arr; }
    function clear(node){ if(!node) return; while(node.firstChild) node.removeChild(node.firstChild); }
    function updateScore(c,t){ refs().score.textContent = `Score: ${c} / ${t}`; }
    function resetMarks(){ $$('.correct,.wrong').forEach(x=>x.classList.remove('correct','wrong')); }
  
    function makeDraggable(el){
      el.setAttribute('draggable','true');
      el.addEventListener('dragstart', e=>{
        const parent = el.parentElement;
        let fromType = '';
        let fromIndex = -1;
  
        if(parent && parent.classList.contains('vj-p-slot')){
          fromType = 'palette-slot';
          fromIndex = parseInt(parent.dataset.slotIndex, 10);
        } else if(parent && parent.classList.contains('vj-cell')){
          fromType = 'grid-cell';
          fromIndex = parseInt(parent.dataset.cellIndex, 10);
        } else if(parent && parent.classList.contains('vj-slot')){
          fromType = 'video-timeline';
        } else if(parent && parent.classList.contains('vj-line-slot')){
          fromType = 'depth-line';
        }
  
        e.dataTransfer.setData('text/id', el.dataset.id);
        e.dataTransfer.setData('text/fromType', fromType);
        e.dataTransfer.setData('text/fromIndex', String(fromIndex));
      });
    }
  
    function makeDropZone(el, onDrop){
      el.addEventListener('dragover', e=>{ e.preventDefault(); el.classList.add('drop-ok'); });
      el.addEventListener('dragleave', ()=> el.classList.remove('drop-ok'));
      el.addEventListener('drop', e=>{
        e.preventDefault(); el.classList.remove('drop-ok');
        const id = e.dataTransfer.getData('text/id');
        if(!id) return;
        onDrop(e, id, el);
      });
    }
  
    // --------- Image mode (slot-based palette) ---------
    function createImgTile(id, src){
      const d = document.createElement('div');
      d.className = 'vj-tile'; d.dataset.id = String(id);
      const img = document.createElement('img'); img.src = src; img.alt='tile';
      d.appendChild(img);
      makeDraggable(d);
      return d;
    }
  
    function createPaletteSlot(idx){
      const slot = document.createElement('div');
      slot.className = 'vj-p-slot';        // square slot in palette
      slot.dataset.slotIndex = String(idx);
      // each slot is its own drop zone
      makeDropZone(slot, (e, id, targetSlot)=>{
        const fromType  = e.dataTransfer.getData('text/fromType');
        const fromIndex = parseInt(e.dataTransfer.getData('text/fromIndex') || '-1', 10);
        const dragged   = document.querySelector(`.vj-tile[data-id="${id}"]`);
        if(!dragged) return;
  
        const targetChild = targetSlot.firstChild; // might be null
  
        if(fromType === 'palette-slot'){
          // swap between slots
          const originSlot = refs().paletteImg.querySelector(`.vj-p-slot[data-slot-index="${fromIndex}"]`);
          if(!originSlot) return;
          if(targetChild) originSlot.appendChild(targetChild);
          targetSlot.appendChild(dragged);
        } else if(fromType === 'grid-cell'){
          // swap with grid cell content
          const originCell = refs().gridImg.querySelector(`.vj-cell[data-cell-index="${fromIndex}"]`);
          if(!originCell) return;
          if(targetChild) originCell.appendChild(targetChild);
          targetSlot.appendChild(dragged);
        }
        // other types ignored for image mode
      });
      return slot;
    }
  
    function createImgCell(expectedId, cellIndex){
      const d = document.createElement('div');
      d.className = 'vj-cell';
      d.dataset.expectedId = String(expectedId);
      d.dataset.cellIndex  = String(cellIndex);
      makeDropZone(d, (e, id, targetCell)=>{
        const fromType  = e.dataTransfer.getData('text/fromType');
        const fromIndex = parseInt(e.dataTransfer.getData('text/fromIndex') || '-1', 10);
        const dragged   = document.querySelector(`.vj-tile[data-id="${id}"]`);
        if(!dragged) return;
  
        const targetChild = targetCell.firstChild;
  
        if(fromType === 'palette-slot'){
          // swap with palette slot
          const originSlot = refs().paletteImg.querySelector(`.vj-p-slot[data-slot-index="${fromIndex}"]`);
          if(!originSlot) return;
          if(targetChild) originSlot.appendChild(targetChild);
          targetCell.appendChild(dragged);
        } else if(fromType === 'grid-cell'){
          // swap grid cell <-> grid cell
          const originCell = refs().gridImg.querySelector(`.vj-cell[data-cell-index="${fromIndex}"]`);
          if(!originCell) return;
          if(targetChild) originCell.appendChild(targetChild);
          targetCell.appendChild(dragged);
        }
      });
      return d;
    }
  
    function sliceImage(img, n){
      const w = img.naturalWidth, h = img.naturalHeight;
      const tw = Math.floor(w/n), th = Math.floor(h/n);
      const cvs = document.createElement('canvas'), ctx = cvs.getContext('2d');
      const tiles = []; let id = 0;
      for(let r=0;r<n;r++){
        for(let c=0;c<n;c++){
          cvs.width = tw; cvs.height = th;
          ctx.drawImage(img, c*tw, r*th, tw, th, 0, 0, tw, th);
          tiles.push({ id: id++, src: cvs.toDataURL('image/jpeg', 0.9) });
        }
      }
      return { tiles, ar: h / w };
    }
  
    function buildImage(tiles, ar){
      S.originalTiles = tiles.map(t=>({...t}));
      S.answerOrder = tiles.map(t=>t.id);
      const R = refs();
      clear(R.paletteImg); clear(R.gridImg);
  
      setCols(R.paletteImg, S.gridN);
      setCols(R.gridImg,    S.gridN);
      setAR(R.paletteImg, ar);
      setAR(R.gridImg,    ar);
  
      // Grid cells with deterministic cellIndex (0..N*N-1)
      // Use the same traversal order as answerOrder to make cellIndex stable
      let cellIdx = 0;
      for(const id of S.answerOrder){
        R.gridImg.appendChild(createImgCell(id, cellIdx++));
      }
  
      // Palette slots (fixed count N*N)
      const total = S.gridN * S.gridN;
      for(let i=0;i<total;i++){
        R.paletteImg.appendChild(createPaletteSlot(i));
      }
  
      // Distribute shuffled tiles into slots (one per slot, in order)
      const shuf = shuffle(tiles.slice());
      const slots = $$('.vj-p-slot', R.paletteImg);
      shuf.forEach((t, i)=>{
        const tile = createImgTile(t.id, t.src);
        slots[i].appendChild(tile);
      });
  
      updateScore(0, total);
      resetMarks();
    }
  
    function loadLocalImage(src){
      const abs = new URL(src, location.href).href;
      console.log('[VJ] loading image:', src, '=>', abs);
      const img = new Image();
      img.onload = ()=>{
        const { tiles, ar } = sliceImage(img, S.gridN);
        buildImage(tiles, ar);
      };
      img.onerror = (e)=>{
        console.error('[VJ] Failed to load image:', src, e);
        alert('Failed to load image: ' + src + '\nCheck path and local server.');
        const n = S.gridN, total = n*n;
        // empty slots/grid with AR=1 so UI is visible
        const tiles = Array.from({length: total}, (_,i)=>({id:i,src:''}));
        buildImage(tiles, 1);
      };
      img.src = src;
    }
  
    // --------- Video mode / 3D (unchanged behavior) ---------
    function createVideoClip(id, src){
      const wrap = document.createElement('div'); wrap.className = 'vj-video vj-tile'; wrap.dataset.id = String(id);
      const v = document.createElement('video');
      v.src = src; v.preload='metadata'; v.muted=true; v.loop=true; v.autoplay=true; v.playsInline=true;
      wrap.appendChild(v);
      makeDraggable(wrap);
      return wrap;
    }
    function createVideoSlot(expectedId){
      const d = document.createElement('div'); d.className = 'vj-slot'; d.dataset.expectedId = String(expectedId);
      makeDropZone(d, (e, id, el)=>{
        if(el.firstChild){ refs().paletteVideo.appendChild(el.firstChild); }
        const clip = document.querySelector(`.vj-video[data-id="${id}"]`);
        if(clip) el.appendChild(clip);
      });
      return d;
    }
    function buildVideo(ar=9/16){
      const R = refs();
      clear(R.paletteVideo); clear(R.timeline);
      setCols(R.paletteVideo, 6); setCols(R.timeline, 6);
      setAR(R.paletteVideo, ar);
      setAR(R.timeline, ar);
      for(let i=0;i<6;i++) R.timeline.appendChild(createVideoSlot(i));
      shuffle(S.videoClips.slice()).forEach(c=> R.paletteVideo.appendChild(createVideoClip(c.id, c.src)));
      updateScore(0, 6); resetMarks();
      const firstVid = R.paletteVideo.querySelector('video');
      if(firstVid){
        firstVid.addEventListener('loadedmetadata', ()=>{
          const ar2 = firstVid.videoHeight / firstVid.videoWidth;
          setAR(R.paletteVideo, ar2); setAR(R.timeline, ar2);
        }, { once:true });
      }
    }
  
    function createDepthToken(k){
      const d = document.createElement('div'); d.className='vj-token vj-tile'; d.dataset.id = String(k);
      d.textContent = String(k);
      makeDraggable(d);
      return d;
    }
    function createLineSlot(expectedId){
      const d = document.createElement('div'); d.className='vj-line-slot'; d.dataset.expectedId = String(expectedId);
      makeDropZone(d, (e, id, el)=>{
        if(el.firstChild){ refs().palette3D.appendChild(el.firstChild); }
        const tok = document.querySelector(`.vj-token[data-id="${id}"]`);
        if(tok) el.appendChild(tok);
      });
      return d;
    }
    function build3D(ar = 1){
        const R = refs();
        clear(R.palette3D); clear(R.line3D);
        setCols(R.palette3D, 6); setCols(R.line3D, 6);
        setAR(R.palette3D, ar); setAR(R.line3D, ar);
      
        // slots for the answer order (near → far)
        for (let i = 0; i < 6; i++) {
          R.line3D.appendChild(createLineSlot(S.depthOrderCorrect[i]));
        }
      
        // === Fixed order tokens: 1..6 (no shuffle) ===
        S.depthTokens = [1, 2, 3, 4, 5, 6];
        S.depthTokens.forEach(k => R.palette3D.appendChild(createDepthToken(k)));
      
        updateScore(0, 6); resetMarks();
      
        if (S.threeDImage) {
          R.threeDImage.onload = () => {
            const ar2 = R.threeDImage.naturalHeight / R.threeDImage.naturalWidth;
            setAR(R.palette3D, ar2); setAR(R.line3D, ar2);
          };
          R.threeDImage.src = S.threeDImage;
        }
      }
  
    // --------- Actions ---------
    function doCheck(){
      resetMarks();
      if(S.mode==='image'){
        const cells = $$('.vj-cell', refs().gridImg);
        let ok=0; cells.forEach(c=>{
          const exp=c.dataset.expectedId, t=c.firstChild;
          (t && t.dataset.id===exp) ? (ok++, c.classList.add('correct')) : c.classList.add('wrong');
        });
        updateScore(ok, cells.length);
      }else if(S.mode==='video'){
        const slots = $$('.vj-slot', refs().timeline);
        let ok=0; slots.forEach(s=>{
          const exp=s.dataset.expectedId, t=s.firstChild;
          (t && t.dataset.id===exp) ? (ok++, s.classList.add('correct')) : s.classList.add('wrong');
        });
        updateScore(ok, 6);
      }else{
        const slots = $$('.vj-line-slot', refs().line3D);
        let ok=0; slots.forEach(s=>{
          const exp=s.dataset.expectedId, t=s.firstChild;
          (t && t.dataset.id===exp) ? (ok++, s.classList.add('correct')) : s.classList.add('wrong');
        });
        updateScore(ok, 6);
      }
    }
  
    function doReset(){
      if(S.mode==='image'){ if(S.imageSrc) loadLocalImage(S.imageSrc); }
      else if(S.mode==='video'){ buildVideo(); }
      else { build3D(); }
    }
  
    function doShuffle(){
      if(S.mode==='image'){
        // 只打乱 Palette 中的 tile，slot 本身不动
        const pal = refs().paletteImg;
        const slots = $$('.vj-p-slot', pal);
        // 收集 tiles
        const tiles = slots.map(s=> s.firstChild).filter(Boolean);
        shuffle(tiles);
        // 清空所有 slot，再按顺序填回
        slots.forEach(s=> s.firstChild && s.removeChild(s.firstChild));
        tiles.forEach((t,i)=> slots[i].appendChild(t));
      }else if(S.mode==='video'){
        const R=refs();
        const tiles=$$('.vj-video', R.paletteVideo);
        shuffle(tiles); clear(R.paletteVideo); tiles.forEach(t=> R.paletteVideo.appendChild(t));
      }else{
        const R=refs();
        const tiles=$$('.vj-token', R.palette3D);
        shuffle(tiles); clear(R.palette3D); tiles.forEach(t=> R.palette3D.appendChild(t));
      }
    }
  
    function switchMode(tab){
      S.mode = (tab==='view3d') ? 'view3d' : tab;
      $$('#vj-tabs li').forEach(x=>x.classList.remove('is-active'));
      $(`#vj-tabs li[data-tab="${tab}"]`).classList.add('is-active');
      $$('#vj-panels .vj-panel').forEach(p=> p.classList.toggle('is-hidden', p.getAttribute('data-panel')!==tab));
      $$('[data-for="image"]').forEach(el=> el.style.display = (S.mode==='image') ? '' : 'none');
  
      if(S.mode==='image'){
        if(S.imageSrc) loadLocalImage(S.imageSrc);
        else {
          const n=S.gridN, total=n*n;
          const tiles = Array.from({length: total}, (_,i)=>({id:i,src:''}));
          buildImage(tiles, 1);
        }
      } else if(S.mode==='video'){
        buildVideo();
      } else {
        build3D();
      }
      const shuffleBtn = refs().btnShuffle;
        if (S.mode === 'view3d') {
        shuffleBtn.disabled = true;
        shuffleBtn.title = 'Shuffle is disabled for 3D depth ordering';
        } else {
        shuffleBtn.disabled = false;
        shuffleBtn.removeAttribute('title');
        }
    }
  
    // --------- Public API ---------
    window.VJPlayground = {
      init(opts={}){
        S.imageSrc      = opts.imageSrc || null;
        S.threeDImage   = opts.threeDImage || null;
        if(Array.isArray(opts.threeDCorrectOrder) && opts.threeDCorrectOrder.length===6){
          S.depthOrderCorrect = opts.threeDCorrectOrder.slice();
        }
  
        // tabs
        $$('#vj-tabs li').forEach(li=>{
          li.addEventListener('click', ()=> switchMode(li.getAttribute('data-tab')));
        });
  
        // buttons
        refs().btnCheck.addEventListener('click', doCheck);
        refs().btnReset.addEventListener('click', doReset);
        refs().btnShuffle.addEventListener('click', doShuffle);
  
        // image grid size
        refs().gridSizeSel.addEventListener('change', e=>{
          S.gridN = parseInt(e.target.value,10);
          if(S.imageSrc) loadLocalImage(S.imageSrc);
        });
  
        // boot
        switchMode('image');
        refs().hint.style.display = 'block';
        // refs().hint.innerHTML = 'Image/Video/3D tile aspect ratios follow the source media.';
        if(S.imageSrc) loadLocalImage(S.imageSrc);
      },
  
      // Provide exactly 6 video URLs
      loadVideoClips(urls){
        if(!Array.isArray(urls) || urls.length!==6){
          console.warn('VJPlayground.loadVideoClips: need exactly 6 clips');
          return;
        }
        S.videoClips = urls.map((u,i)=>({id:i, src:u}));
        if(S.mode==='video') buildVideo();
      },
  
      setVideoAnswer(order){
        if(order && order.length===6) S.videoAnswer = order.slice();
      },
  
      set3DImage(url){
        S.threeDImage = url;
        if(refs().threeDImage) refs().threeDImage.src = url;
      },
  
      set3DCorrectOrder(order){
        if(order && order.length===6){
          S.depthOrderCorrect = order.slice();
          if(S.mode==='view3d') build3D();
        }
      }
    };
  })();
  