package com.cupet.fallingball.learning

import android.annotation.SuppressLint
import android.graphics.Color
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import com.cupet.fallingball.ui.UiColors
import org.json.JSONObject
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Local-only rich text renderer. The document is loaded once, then updated through a
 * requestAnimationFrame-coalesced JavaScript renderer. This keeps completed Markdown blocks
 * alive while only the changing tail is replaced during a response stream.
 */
@SuppressLint("SetJavaScriptEnabled", "AddJavascriptInterface")
internal class LearningRichTextView(
    context: android.content.Context,
    private val onHeightChanged: (heightPx: Int, renderToken: Long) -> Unit
) : WebView(context) {
    private var lastMarkdown = ""
    private var pageLoaded = false
    private var pageLoading = false
    private var pendingMarkdown: String? = null
    private var renderToken = 0L
    private var pageLoadCount = 0

    init {
        setBackgroundColor(Color.TRANSPARENT)
        isVerticalScrollBarEnabled = false
        isHorizontalScrollBarEnabled = false
        overScrollMode = OVER_SCROLL_NEVER
        settings.javaScriptEnabled = true
        settings.allowFileAccess = true
        settings.allowContentAccess = false
        settings.blockNetworkLoads = true
        addJavascriptInterface(HeightBridge(), "AndroidBridge")
        webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean =
                true

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                pageLoading = false
                pageLoaded = true
                pendingMarkdown?.let { markdown ->
                    pendingMarkdown = null
                    dispatchMarkdown(markdown, renderToken)
                } ?: evaluateJavascript("window.__notifyHeight && window.__notifyHeight();", null)
            }
        }
    }

    /**
     * Queues a new source for the existing document. Duplicate sources are ignored only after
     * the page is ready; a source queued while the page is loading is still rendered once.
     */
    fun setMarkdown(markdown: String): Boolean {
        if (markdown == lastMarkdown && pageLoaded && pendingMarkdown == null) return false
        lastMarkdown = markdown
        renderToken += 1L
        pendingMarkdown = markdown
        if (pageLoaded) {
            pendingMarkdown = null
            dispatchMarkdown(markdown, renderToken)
        } else if (!pageLoading) {
            loadPage()
        }
        return true
    }

    /** Invalidates callbacks from a previous RecyclerView binding. */
    fun invalidateRenderToken() {
        renderToken += 1L
        pendingMarkdown = null
        lastMarkdown = ""
    }

    fun currentRenderToken(): Long = renderToken

    internal fun currentPageLoadCount(): Int = pageLoadCount

    internal fun acceptHeightMeasurement(cssPixels: Int, token: Long): Boolean {
        if (token != renderToken) return false
        val density = resources.displayMetrics.density
        val boundedCssPixels = cssPixels.coerceIn(1, MAX_HEIGHT_CSS_PX)
        val height = (boundedCssPixels * density).roundToInt()
            .coerceAtLeast((MIN_HEIGHT_DP * density).roundToInt())
        onHeightChanged(height, token)
        return true
    }

    private fun loadPage() {
        if (pageLoading) return
        pageLoading = true
        pageLoadCount += 1
        loadDataWithBaseURL(
            "file:///android_asset/learning/",
            documentHtml(),
            "text/html",
            "UTF-8",
            null
        )
    }

    private fun dispatchMarkdown(markdown: String, token: Long) {
        val quoted = JSONObject.quote(markdown)
        evaluateJavascript("window.__setMarkdown($quoted,$token);", null)
    }

    private inner class HeightBridge {
        @JavascriptInterface
        fun onHeightChanged(cssPixels: Int, token: Long) {
            post { acceptHeightMeasurement(cssPixels, token) }
        }
    }

    private fun documentHtml(): String {
        val onSurface = css(UiColors.ON_SURFACE)
        val onSurfaceVariant = css(UiColors.ON_SURFACE_VARIANT)
        val surfaceContainer = css(UiColors.SURFACE_CONTAINER)
        val surfaceContainerHigh = css(UiColors.SURFACE_CONTAINER_HIGH)
        val outlineVariant = css(UiColors.OUTLINE_VARIANT)
        val primary = css(UiColors.PRIMARY)
        val onSecondaryContainer = css(UiColors.ON_SECONDARY_CONTAINER)
        val secondaryContainer = css(UiColors.SECONDARY_CONTAINER)
        return """
            <!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
            <link rel="stylesheet" href="katex/katex.min.css">
            <style>
              :root{color-scheme:light}
              html,body{margin:0;padding:0;background:transparent;color:$onSurface;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:15px;line-height:1.55;overflow:hidden}
              body{padding:11px 22px;box-sizing:border-box;word-break:break-word}
              #content> :first-child{margin-top:0!important}
              #content> :last-child{margin-bottom:0!important}
              p{margin:0 0 9px} h1,h2,h3{margin:7px 0 4px;font-size:1.04em} ul,ol{margin:4px 0 8px;padding-left:21px}
              pre{white-space:pre-wrap;background:$surfaceContainer;border:1px solid $outlineVariant;border-radius:7px;padding:9px;font-family:monospace;font-size:12px;overflow-x:auto;color:$onSurface}
              code{background:$surfaceContainerHigh;border-radius:4px;padding:1px 4px;font-family:monospace;font-size:.9em;color:$onSecondaryContainer}
              blockquote{margin:7px 0;padding:2px 10px;border-left:3px solid $primary;color:$onSurfaceVariant}
              table{width:100%;border-collapse:collapse;margin:8px 0;font-size:.93em} th,td{border:1px solid $outlineVariant;padding:5px 7px;text-align:left} th{background:$secondaryContainer;color:$onSecondaryContainer}
              a{color:$primary;text-decoration:underline} img{max-width:100%;height:auto;border-radius:6px}
              .katex-display{overflow-x:auto;overflow-y:hidden;margin:8px 0}
            </style></head><body><div id="content"></div>
            <script src="marked/marked.min.js"></script><script src="katex/katex.min.js"></script>
            <script src="katex/contrib/auto-render.min.js"></script>
            <script>
              (function(){
                const root=document.getElementById('content');
                let queuedSource=''; let queuedToken=0; let frame=0; let latestToken=0;
                function notifyHeight(token){
                  requestAnimationFrame(function(){
                    // body.scrollHeight is never smaller than the current WebView viewport,
                    // so it cannot report a shrink. Measure the Markdown root plus body padding.
                    const h=Math.ceil(Math.max(root.scrollHeight+22,1));
                    if(window.AndroidBridge) AndroidBridge.onHeightChanged(h,token);
                  });
                }
                function reconcile(next){
                  const newNodes=Array.from(next.children);
                  for(let i=0;i<newNodes.length;i++){
                    const oldNode=root.children[i]; const newNode=newNodes[i];
                    if(!oldNode){ root.appendChild(newNode); continue; }
                    if(oldNode.outerHTML!==newNode.outerHTML) root.replaceChild(newNode,oldNode);
                  }
                  while(root.children.length>newNodes.length) root.lastElementChild.remove();
                }
                function render(source,token){
                  const next=document.createElement('div');
                  try{
                    marked.setOptions({gfm:true,breaks:true,pedantic:false});
                    next.innerHTML=marked.parse(source || '');
                    next.querySelectorAll('script,iframe,object,embed,form,input,button,textarea,select,option,datalist,fieldset,legend,canvas,video,audio,source,track,style,link,meta').forEach(function(node){node.remove();});
                    const allowedAttributes={a:['href','title'],img:['src','alt','title'],code:['class'],ol:['start'],td:['colspan','rowspan'],th:['colspan','rowspan']};
                    next.querySelectorAll('*').forEach(function(node){
                      Array.from(node.attributes).forEach(function(attr){
                        const name=attr.name.toLowerCase();
                        const tag=node.tagName.toLowerCase();
                        const allowed=allowedAttributes[tag] || [];
                        if(allowed.indexOf(name)<0){
                          node.removeAttribute(attr.name); return;
                        }
                        if(name==='href' || name==='src'){
                          try{
                            const parsed=new URL(String(attr.value || ''),document.baseURI);
                            if(parsed.protocol!=='https:' && parsed.protocol!=='http:') node.removeAttribute(attr.name);
                          }catch(urlError){node.removeAttribute(attr.name);}
                        }
                      });
                    });
                    renderMathInElement(next,{delimiters:[
                      {left:'$$',right:'$$',display:true},
                      {left:'\\\\[',right:'\\\\]',display:true},
                      {left:'\\\\(',right:'\\\\)',display:false},
                      {left:'$',right:'$',display:false}
                    ],ignoredTags:['script','style','textarea','pre','code'],ignoredClasses:['katex'],throwOnError:false,trust:false,strict:false});
                  }catch(error){
                    const fallback=document.createElement('p');
                    fallback.textContent=source || '';
                    next.innerHTML=''; next.appendChild(fallback);
                  }
                  reconcile(next); latestToken=token; notifyHeight(token);
                }
                window.__setMarkdown=function(source,token){
                  queuedSource=String(source || ''); queuedToken=Number(token || 0);
                  if(frame) return;
                  frame=requestAnimationFrame(function(){frame=0;render(queuedSource,queuedToken);});
                };
                window.__notifyHeight=function(){notifyHeight(latestToken);};
                if(window.ResizeObserver){new ResizeObserver(function(){notifyHeight(latestToken);}).observe(root);}
              })();
            </script></body></html>
        """.trimIndent()
    }

    private fun css(color: Int): String =
        String.format(Locale.ROOT, "#%06X", color and 0xFFFFFF)

    private companion object {
        const val MIN_HEIGHT_DP = 44
        const val MAX_HEIGHT_CSS_PX = 30_000
    }
}
