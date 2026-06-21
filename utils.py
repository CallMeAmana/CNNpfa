import streamlit.components.v1 as components


def inject_sidebar_toggle():
    """Bouton hamburger fixe pour afficher/masquer la sidebar.

    Nécessaire car style.css met le header à height:0, cachant le bouton natif.
    Détecte l'état via getBoundingClientRect (plus fiable que aria-expanded).
    """
    components.html("""
<script>
(function () {
  function addFAB() {
    var doc = window.parent.document;
    if (doc.getElementById('__st_sidebar_fab')) return;

    var fab = doc.createElement('button');
    fab.id    = '__st_sidebar_fab';
    fab.title = 'Afficher / masquer la sidebar';
    fab.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" '
      + 'viewBox="0 0 24 24" fill="none" stroke="#00e5c8" stroke-width="2.5" '
      + 'stroke-linecap="round" stroke-linejoin="round">'
      + '<line x1="3" y1="6"  x2="21" y2="6"/>'
      + '<line x1="3" y1="12" x2="21" y2="12"/>'
      + '<line x1="3" y1="18" x2="21" y2="18"/>'
      + '</svg>';

    fab.style.cssText = [
      'position:fixed', 'top:10px', 'left:10px',
      'z-index:2147483647',
      'width:32px', 'height:32px',
      'background:#0b1729',
      'border:1px solid rgba(0,229,200,0.28)',
      'border-radius:7px',
      'cursor:pointer',
      'display:flex', 'align-items:center', 'justify-content:center',
      'padding:0',
      'box-shadow:0 2px 12px rgba(0,0,0,0.6)',
      'transition:background 0.15s,border-color 0.15s,box-shadow 0.15s'
    ].join(';');

    fab.onmouseenter = function () {
      this.style.background  = 'rgba(0,229,200,0.08)';
      this.style.borderColor = '#00e5c8';
      this.style.boxShadow   = '0 0 14px rgba(0,229,200,0.25)';
    };
    fab.onmouseleave = function () {
      this.style.background  = '#0b1729';
      this.style.borderColor = 'rgba(0,229,200,0.28)';
      this.style.boxShadow   = '0 2px 12px rgba(0,0,0,0.6)';
    };

    fab.onclick = function () {
      var doc = window.parent.document;

      /* ── détecter l'état réel de la sidebar via sa largeur ── */
      var sb = doc.querySelector('[data-testid="stSidebar"]');
      var sbWidth = sb ? sb.getBoundingClientRect().width : 0;
      var isOpen  = sbWidth > 50;

      if (isOpen) {
        /* sidebar ouverte → chercher le bouton pour la fermer */
        var closeBtn =
          doc.querySelector('[data-testid="stSidebarCollapseButton"]') ||
          doc.querySelector('[data-testid="stSidebar"] button[kind="header"]') ||
          (sb ? sb.querySelector('button') : null);
        if (closeBtn) { closeBtn.click(); return; }

      } else {
        /* sidebar fermée → chercher le bouton pour l'ouvrir */
        var openBtn =
          doc.querySelector('[data-testid="collapsedControl"]') ||
          doc.querySelector('button[data-testid="collapsedControl"]') ||
          doc.querySelector('[data-testid="stHeader"] button') ||
          doc.querySelector('header button');
        if (openBtn) { openBtn.click(); return; }

        /* ── fallback : forcer l'affichage CSS directement ── */
        if (sb) {
          sb.style.setProperty('width',      '260px',  'important');
          sb.style.setProperty('min-width',  '260px',  'important');
          sb.style.setProperty('transform',  'none',   'important');
          sb.style.setProperty('visibility', 'visible','important');
          sb.style.setProperty('display',    'flex',   'important');
          sb.setAttribute('aria-expanded', 'true');

          /* repousser le contenu principal */
          var main = doc.querySelector('[data-testid="stMain"]') ||
                     doc.querySelector('.main');
          if (main) {
            main.style.setProperty('margin-left', '260px', 'important');
          }
        }
      }
    };

    doc.body.appendChild(fab);
  }

  addFAB();
  setTimeout(addFAB, 300);
  setTimeout(addFAB, 1000);
})();
</script>
""", height=0, scrolling=False)
