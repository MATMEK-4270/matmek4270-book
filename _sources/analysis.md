 # Lecture 2

<section id="recap---finite-differencing-of-exponential-decay" class="slide level2 smaller">
<h2 class="smaller">Recap - Finite differencing of exponential decay</h2>
<div class="hidden">

</div>
<div class="callout callout-note callout-titled callout-style-default">
<div class="callout-body">
<div class="callout-title">
<div class="callout-icon-container">
<i class='callout-icon'></i>
</div>
<p><strong>The ordinary differential equation</strong></p>
</div>
<div class="callout-content">
<p><span class="math display">\[
u&#39;(t) = -au(t),\quad u(0)=I, \quad y \in (0, T]
\]</span> where <span class="math inline">\(a&gt;0\)</span> is a constant.</p>
</div>
</div>
</div>
<p>Solve the ODE by finite difference methods:</p>
<ul>
<li><p>Discretize in time:</p>
<p><span class="math display">\[0 = t_0 &lt; t_1 &lt; t_2 &lt; \cdots &lt; t_{N_t-1} &lt; t_{N_t} = T\]</span></p></li>
<li><p>Satisfy the ODE at <span class="math inline">\(N_t\)</span> discrete time steps:</p>
<p><span class="math display">\[
\begin{align}
u&#39;(t_n) &amp;= -a u(t_n), \quad &amp;n\in [1, \ldots, N_t], \text{ or} \\
u&#39;(t_{n+\scriptstyle\frac{1}{2}}) &amp;= -a u(t_{n+\scriptstyle\frac{1}{2}}), \quad &amp;n\in [0, \ldots, N_t-1]
\end{align}
\]</span></p></li>
</ul>
</section>
<section id="finite-difference-algorithms" class="slide level2 smaller">
<h2 class="smaller">Finite difference algorithms</h2>
<ul>
<li>Discretization by a generic <span class="math inline">\(\theta\)</span>-rule</li>
</ul>
<p><span class="math display">\[
\frac{u^{n+1}-u^{n}}{\triangle t} = -(1-\theta)au^{n} - \theta u^{n+1}
\]</span></p>
<p><span class="math display">\[
\begin{cases}
  \theta = 0 \quad &amp;\text{Forward Euler} \\
  \theta = 1 \quad &amp;\text{Backward Euler} \\
  \theta = 1/2 \quad &amp;\text{Crank-Nicolson}
  \end{cases}
\]</span></p>
<p>Note <span class="math inline">\(u^n = u(t_n)\)</span></p>
<ul>
<li>Solve recursively: Set <span class="math inline">\(u^0 = I\)</span> and then</li>
</ul>
<p><span class="math display">\[
u^{n+1} = \frac{1-(1-\theta)a \triangle t}{1+\theta a \triangle t}u^{n} \quad \text{for } n=0, 1, \ldots
\]</span></p>
</section>
<section id="analysis-of-finite-difference-equations" class="slide level2">
<h2>Analysis of finite difference equations</h2>
<p>Model: <span class="math display">\[
u&#39;(t) = -au(t),\quad u(0)=I
\]</span></p>
<p>Method:</p>
<p><span class="math display">\[
u^{n+1} = \frac{1 - (1-\theta) a\Delta t}{1 + \theta a\Delta t}u^n
\]</span></p>
<div class="callout callout-note callout-titled callout-style-default">
<div class="callout-body">
<div class="callout-title">
<div class="callout-icon-container">
<i class='callout-icon'></i>
</div>
<p><strong>Problem setting</strong></p>
</div>
<div class="callout-content">
<p>How good is this method? Is it safe to use it?</p>
</div>
</div>
</div>
</section>
<section id="encouraging-numerical-solutions---backwards-euler" class="slide level2">
<h2>Encouraging numerical solutions - Backwards Euler</h2>
<p><span class="math inline">\(I=1\)</span>, <span class="math inline">\(a=2\)</span>, <span class="math inline">\(\theta =1\)</span>, <span class="math inline">\(\Delta t=1.25, 0.75, 0.5, 0.1\)</span>.</p>
<div class="sourceCode" id="cb1"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> matplotlib.pyplot <span class="im">as</span> plt</span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> numpy <span class="im">as</span> np</span>
<span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> solver(I, a, T, dt, theta):</span>
<span id="cb1-4"><a href="#cb1-4" aria-hidden="true" tabindex="-1"></a>    <span class="co">&quot;&quot;&quot;Solve u&#39;=-a*u, u(0)=I, for t in (0, T] with steps of dt.&quot;&quot;&quot;</span></span>
<span id="cb1-5"><a href="#cb1-5" aria-hidden="true" tabindex="-1"></a>    Nt <span class="op">=</span> <span class="bu">int</span>(T<span class="op">/</span>dt)            <span class="co"># no of time intervals</span></span>
<span id="cb1-6"><a href="#cb1-6" aria-hidden="true" tabindex="-1"></a>    T <span class="op">=</span> Nt<span class="op">*</span>dt                 <span class="co"># adjust T to fit time step dt</span></span>
<span id="cb1-7"><a href="#cb1-7" aria-hidden="true" tabindex="-1"></a>    u <span class="op">=</span> np.zeros(Nt<span class="op">+</span><span class="dv">1</span>)           <span class="co"># array of u[n] values</span></span>
<span id="cb1-8"><a href="#cb1-8" aria-hidden="true" tabindex="-1"></a>    t <span class="op">=</span> np.linspace(<span class="dv">0</span>, T, Nt<span class="op">+</span><span class="dv">1</span>)  <span class="co"># time mesh</span></span>
<span id="cb1-9"><a href="#cb1-9" aria-hidden="true" tabindex="-1"></a>    u[<span class="dv">0</span>] <span class="op">=</span> I                  <span class="co"># assign initial condition</span></span>
<span id="cb1-10"><a href="#cb1-10" aria-hidden="true" tabindex="-1"></a>    u[<span class="dv">1</span>:] <span class="op">=</span> (<span class="dv">1</span> <span class="op">-</span> (<span class="dv">1</span><span class="op">-</span>theta)<span class="op">*</span>a<span class="op">*</span>dt)<span class="op">/</span>(<span class="dv">1</span> <span class="op">+</span> theta<span class="op">*</span>dt<span class="op">*</span>a)</span>
<span id="cb1-11"><a href="#cb1-11" aria-hidden="true" tabindex="-1"></a>    u[:] <span class="op">=</span> np.cumprod(u)</span>
<span id="cb1-12"><a href="#cb1-12" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> u, t</span>
<span id="cb1-13"><a href="#cb1-13" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb1-14"><a href="#cb1-14" aria-hidden="true" tabindex="-1"></a>u_exact <span class="op">=</span> <span class="kw">lambda</span> I, a, t: I<span class="op">*</span>np.exp(<span class="op">-</span>a<span class="op">*</span>t)</span>
<span id="cb1-15"><a href="#cb1-15" aria-hidden="true" tabindex="-1"></a>I, a, T, theta <span class="op">=</span> <span class="dv">1</span>, <span class="dv">2</span>, <span class="dv">8</span>, <span class="dv">1</span></span>
<span id="cb1-16"><a href="#cb1-16" aria-hidden="true" tabindex="-1"></a>dt <span class="op">=</span> np.array([<span class="fl">1.25</span>, <span class="fl">0.75</span>, <span class="fl">0.5</span>, <span class="fl">0.1</span>])</span>
<span id="cb1-17"><a href="#cb1-17" aria-hidden="true" tabindex="-1"></a>fig, axs <span class="op">=</span> plt.subplots(<span class="dv">2</span>, <span class="dv">2</span>)</span>
<span id="cb1-18"><a href="#cb1-18" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb1-19"><a href="#cb1-19" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> j <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb1-20"><a href="#cb1-20" aria-hidden="true" tabindex="-1"></a>        u0, t0 <span class="op">=</span> solver(I, a, T, dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j], theta)</span>
<span id="cb1-21"><a href="#cb1-21" aria-hidden="true" tabindex="-1"></a>        axs[i, j].plot(t0, u0, <span class="st">&#39;b&#39;</span>, t0, u_exact(I, a, t0), <span class="st">&#39;r--&#39;</span>)</span>
<span id="cb1-22"><a href="#cb1-22" aria-hidden="true" tabindex="-1"></a>        axs[i, j].legend([<span class="st">&#39;numerical&#39;</span>, <span class="st">&#39;exact&#39;</span>])</span>
<span id="cb1-23"><a href="#cb1-23" aria-hidden="true" tabindex="-1"></a>        axs[i, j].set_title(<span class="ss">f&#39;$\Delta t = </span><span class="sc">{</span>dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j]<span class="sc">}</span><span class="ss">$&#39;</span>)</span>
<span id="cb1-24"><a href="#cb1-24" aria-hidden="true" tabindex="-1"></a>        axs[i, j].label_outer()</span></code></pre></div>
<pre><code>&lt;&gt;:23: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

&lt;&gt;:23: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

/var/folders/ql/nly59sfj4dd3spkth6lk0kjw0000gn/T/ipykernel_33739/3023454349.py:23: SyntaxWarning:

invalid escape sequence &#39;\D&#39;
</code></pre>
<p><img data-src="analysis_files/figure-revealjs/cell-2-output-2.png" /></p>
</section>
<section id="discouraging-numerical-solutions---crank-nicolson" class="slide level2">
<h2>Discouraging numerical solutions - Crank-Nicolson</h2>
<p><span class="math inline">\(I=1\)</span>, <span class="math inline">\(a=2\)</span>, <span class="math inline">\(\theta=0.5\)</span>, <span class="math inline">\(\Delta t=1.25, 0.75, 0.5, 0.1\)</span>.</p>
<div class="sourceCode" id="cb3"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb3-1"><a href="#cb3-1" aria-hidden="true" tabindex="-1"></a>I, a, T, theta <span class="op">=</span> <span class="dv">1</span>, <span class="dv">2</span>, <span class="dv">8</span>, <span class="fl">0.5</span></span>
<span id="cb3-2"><a href="#cb3-2" aria-hidden="true" tabindex="-1"></a>fig, axs <span class="op">=</span> plt.subplots(<span class="dv">2</span>, <span class="dv">2</span>)</span>
<span id="cb3-3"><a href="#cb3-3" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb3-4"><a href="#cb3-4" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> j <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb3-5"><a href="#cb3-5" aria-hidden="true" tabindex="-1"></a>        u0, t0 <span class="op">=</span> solver(I, a, T, dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j], theta)</span>
<span id="cb3-6"><a href="#cb3-6" aria-hidden="true" tabindex="-1"></a>        axs[i, j].plot(t0, u0, <span class="st">&#39;b&#39;</span>, t0, u_exact(I, a, t0), <span class="st">&#39;r--&#39;</span>)</span>
<span id="cb3-7"><a href="#cb3-7" aria-hidden="true" tabindex="-1"></a>        axs[i, j].legend([<span class="st">&#39;numerical&#39;</span>, <span class="st">&#39;exact&#39;</span>])</span>
<span id="cb3-8"><a href="#cb3-8" aria-hidden="true" tabindex="-1"></a>        axs[i, j].set_title(<span class="ss">f&#39;$\Delta t = </span><span class="sc">{</span>dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j]<span class="sc">}</span><span class="ss">$&#39;</span>)</span>
<span id="cb3-9"><a href="#cb3-9" aria-hidden="true" tabindex="-1"></a>        axs[i, j].label_outer()</span></code></pre></div>
<pre><code>&lt;&gt;:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

&lt;&gt;:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

/var/folders/ql/nly59sfj4dd3spkth6lk0kjw0000gn/T/ipykernel_33739/3714792576.py:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;
</code></pre>
<p><img data-src="analysis_files/figure-revealjs/cell-3-output-2.png" /></p>
</section>
<section id="discouraging-numerical-solutions---forward-euler" class="slide level2">
<h2>Discouraging numerical solutions - Forward Euler</h2>
<p><span class="math inline">\(I=1\)</span>, <span class="math inline">\(a=2\)</span>, <span class="math inline">\(\theta=0\)</span>, <span class="math inline">\(\Delta t=1.25, 0.75, 0.5, 0.1\)</span>.</p>
<div class="sourceCode" id="cb5"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb5-1"><a href="#cb5-1" aria-hidden="true" tabindex="-1"></a>I, a, T, theta <span class="op">=</span> <span class="dv">1</span>, <span class="dv">2</span>, <span class="dv">8</span>, <span class="dv">0</span></span>
<span id="cb5-2"><a href="#cb5-2" aria-hidden="true" tabindex="-1"></a>fig, axs <span class="op">=</span> plt.subplots(<span class="dv">2</span>, <span class="dv">2</span>)</span>
<span id="cb5-3"><a href="#cb5-3" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb5-4"><a href="#cb5-4" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> j <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb5-5"><a href="#cb5-5" aria-hidden="true" tabindex="-1"></a>        u0, t0 <span class="op">=</span> solver(I, a, T, dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j], theta)</span>
<span id="cb5-6"><a href="#cb5-6" aria-hidden="true" tabindex="-1"></a>        axs[i, j].plot(t0, u0, <span class="st">&#39;b&#39;</span>, t0, u_exact(I, a, t0), <span class="st">&#39;r--&#39;</span>)</span>
<span id="cb5-7"><a href="#cb5-7" aria-hidden="true" tabindex="-1"></a>        axs[i, j].legend([<span class="st">&#39;numerical&#39;</span>, <span class="st">&#39;exact&#39;</span>])</span>
<span id="cb5-8"><a href="#cb5-8" aria-hidden="true" tabindex="-1"></a>        axs[i, j].set_title(<span class="ss">f&#39;$\Delta t = </span><span class="sc">{</span>dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j]<span class="sc">}</span><span class="ss">$&#39;</span>)</span>
<span id="cb5-9"><a href="#cb5-9" aria-hidden="true" tabindex="-1"></a>        axs[i, j].label_outer()</span></code></pre></div>
<pre><code>&lt;&gt;:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

&lt;&gt;:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

/var/folders/ql/nly59sfj4dd3spkth6lk0kjw0000gn/T/ipykernel_33739/1931836584.py:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;
</code></pre>
<p><img data-src="analysis_files/figure-revealjs/cell-4-output-2.png" /></p>
</section>
<section id="summary-of-observations" class="slide level2 smaller">
<h2 class="smaller">Summary of observations</h2>
<p>The characteristics of the displayed curves can be summarized as follows:</p>
<ul>
<li>The Backward Euler scheme <em>always</em> gives a monotone solution, lying above the exact solution.</li>
<li>The Crank-Nicolson scheme gives the most accurate results, but for <span class="math inline">\(\Delta t=1.25\)</span> the solution oscillates.</li>
<li>The Forward Euler scheme gives a growing, oscillating solution for <span class="math inline">\(\Delta t=1.25\)</span>; a decaying, oscillating solution for <span class="math inline">\(\Delta t=0.75\)</span>; a strange solution <span class="math inline">\(u^n=0\)</span> for <span class="math inline">\(n\geq 1\)</span> when <span class="math inline">\(\Delta t=0.5\)</span>; and a solution seemingly as accurate as the one by the Backward Euler scheme for <span class="math inline">\(\Delta t = 0.1\)</span>, but the curve lies <em>below</em> the exact solution.</li>
<li>Small enough <span class="math inline">\(\Delta t\)</span> gives stable and accurate solution for all methods!</li>
</ul>
</section>
<section id="problem-setting-1" class="slide level2">
<h2>Problem setting</h2>
<div class="callout callout-note callout-titled callout-style-default">
<div class="callout-body">
<div class="callout-title">
<div class="callout-icon-container">
<i class='callout-icon'></i>
</div>
<p><strong>We ask the question</strong></p>
</div>
<div class="callout-content">
<ul>
<li>Under what circumstances, i.e., values of the input data <span class="math inline">\(I\)</span>, <span class="math inline">\(a\)</span>, and <span class="math inline">\(\Delta t\)</span> will the Forward Euler and Crank-Nicolson schemes result in undesired oscillatory solutions?</li>
</ul>
<p>Techniques of investigation:</p>
<ul>
<li>Numerical experiments</li>
<li>Mathematical analysis</li>
</ul>
<p>Another question to be raised is</p>
<ul>
<li>How does <span class="math inline">\(\Delta t\)</span> impact the error in the numerical solution?</li>
</ul>
</div>
</div>
</div>
</section>
<section id="exact-numerical-solution" class="slide level2 smaller">
<h2 class="smaller">Exact numerical solution</h2>
<p>For the simple exponential decay problem we are lucky enough to have an exact numerical solution</p>
<p><span class="math display">\[
u^{n} = IA^n,\quad A = \frac{1 - (1-\theta) a\Delta t}{1 + \theta a\Delta t}
\]</span></p>
<p>Such a formula for the exact discrete solution is unusual to obtain in practice, but very handy for our analysis here.</p>
<div class="callout callout-note callout-titled callout-style-default">
<div class="callout-body">
<div class="callout-title">
<div class="callout-icon-container">
<i class='callout-icon'></i>
</div>
<p><strong>Note</strong></p>
</div>
<div class="callout-content">
<p>An exact dicrete solution fulfills a discrete equation (without round-off errors), whereas an exact solution fulfills the original mathematical equation.</p>
</div>
</div>
</div>
</section>
<section id="stability" class="slide level2 smaller">
<h2 class="smaller">Stability</h2>
<p>Since <span class="math inline">\(u^n=I A^n\)</span>,</p>
<ul>
<li><span class="math inline">\(A &lt; 0\)</span> gives a factor <span class="math inline">\((-1)^n\)</span> and oscillatory solutions</li>
<li><span class="math inline">\(|A|&gt;1\)</span> gives growing solutions</li>
<li>Recall: the exact solution is <em>monotone</em> and <em>decaying</em></li>
<li>If these qualitative properties are not met, we say that the numerical solution is <em>unstable</em></li>
</ul>
<div class="fragment">
<p>For stability we need</p>
<p><span class="math display">\[
A &gt; 0 \quad \text{ and } \quad |A| \le 1
\]</span></p>
</div>
</section>
<section id="computation-of-stability-in-this-problem" class="slide level2 smaller">
<h2 class="smaller">Computation of stability in this problem</h2>
<p><span class="math inline">\(A &lt; 0\)</span> if</p>
<p><span class="math display">\[
\frac{1 - (1-\theta) a\Delta t}{1 + \theta a\Delta t} &lt; 0
\]</span></p>
<p>To avoid oscillatory solutions we must have <span class="math inline">\(A&gt; 0\)</span>, which happens for</p>
<div class="fragment">
<p><span class="math display">\[
\Delta t &lt; \frac{1}{(1-\theta)a}, \quad \text{for} \, \theta &lt; 1
\]</span></p>
<ul>
<li>Always fulfilled for Backward Euler (<span class="math inline">\(\theta=1 \rightarrow 1 &lt; 1+a \Delta t\)</span> always true)</li>
<li><span class="math inline">\(\Delta t \leq 1/a\)</span> for Forward Euler (<span class="math inline">\(\theta=0\)</span>)</li>
<li><span class="math inline">\(\Delta t \leq 2/a\)</span> for Crank-Nicolson (<span class="math inline">\(\theta = 0.5\)</span>)</li>
</ul>
<p>We get oscillatory solutions for FE when <span class="math inline">\(\Delta t \le 1/a\)</span> and for CN when <span class="math inline">\(\Delta t \le 2/a\)</span></p>
</div>
</section>
<section id="computation-of-stability-in-this-problem-1" class="slide level2 smaller">
<h2 class="smaller">Computation of stability in this problem</h2>
<p><span class="math inline">\(|A|\leq 1\)</span> means <span class="math inline">\(-1\leq A\leq 1\)</span></p>
<p><span class="math display">\[
-1\leq\frac{1 - (1-\theta) a\Delta t}{1 + \theta a\Delta t} \leq 1
\]</span></p>
<ul>
<li><span class="math inline">\(-1\)</span> is the critical limit (because <span class="math inline">\(A\le 1\)</span> is always satisfied).</li>
<li><span class="math inline">\(-1 &lt; A\)</span> is always fulfilled for Backward Euler (<span class="math inline">\(\theta=1\)</span>) and Crank-Nicolson (<span class="math inline">\(\theta=0.5\)</span>).</li>
<li>For forward Euler or simply <span class="math inline">\(\theta &lt; 0.5\)</span> we have <span class="math display">\[
\Delta t \leq \frac{2}{(1-2\theta)a},\quad
\]</span> and thus <span class="math inline">\(\Delta t \leq 2/a\)</span> for stability of the forward Euler (<span class="math inline">\(\theta=0\)</span>) method</li>
</ul>
</section>
<section id="explanation-of-problems-with-forward-euler" class="slide level2 smaller">
<h2 class="smaller">Explanation of problems with forward Euler</h2>
<div class="columns">
<div class="column" style="width:55%;">
<div class="sourceCode" id="cb7"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb7-1"><a href="#cb7-1" aria-hidden="true" tabindex="-1"></a>I, a, T, theta <span class="op">=</span> <span class="dv">1</span>, <span class="dv">2</span>, <span class="dv">8</span>, <span class="dv">0</span></span>
<span id="cb7-2"><a href="#cb7-2" aria-hidden="true" tabindex="-1"></a>fig, axs <span class="op">=</span> plt.subplots(<span class="dv">2</span>, <span class="dv">2</span>)</span>
<span id="cb7-3"><a href="#cb7-3" aria-hidden="true" tabindex="-1"></a>ab <span class="op">=</span> {(<span class="dv">0</span>, <span class="dv">0</span>): <span class="st">&#39;a)&#39;</span>, (<span class="dv">0</span>, <span class="dv">1</span>): <span class="st">&#39;b)&#39;</span>, (<span class="dv">1</span>, <span class="dv">0</span>): <span class="st">&#39;c)&#39;</span>, (<span class="dv">1</span>, <span class="dv">1</span>): <span class="st">&#39;d)&#39;</span>}</span>
<span id="cb7-4"><a href="#cb7-4" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb7-5"><a href="#cb7-5" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> j <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb7-6"><a href="#cb7-6" aria-hidden="true" tabindex="-1"></a>        u0, t0 <span class="op">=</span> solver(I, a, T, dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j], theta)</span>
<span id="cb7-7"><a href="#cb7-7" aria-hidden="true" tabindex="-1"></a>        axs[i, j].plot(t0, u0, <span class="st">&#39;b&#39;</span>, t0, u_exact(I, a, t0), <span class="st">&#39;r--&#39;</span>)</span>
<span id="cb7-8"><a href="#cb7-8" aria-hidden="true" tabindex="-1"></a>        axs[i, j].legend([<span class="st">&#39;numerical&#39;</span>, <span class="st">&#39;exact&#39;</span>])</span>
<span id="cb7-9"><a href="#cb7-9" aria-hidden="true" tabindex="-1"></a>        axs[i, j].set_title(<span class="ss">f&#39;$\Delta t = </span><span class="sc">{</span>dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j]<span class="sc">}</span><span class="ss">$&#39;</span>)</span>
<span id="cb7-10"><a href="#cb7-10" aria-hidden="true" tabindex="-1"></a>        axs[i, j].text(<span class="fl">3.2</span>, u0.<span class="bu">max</span>()<span class="op">*</span><span class="fl">0.85</span>, <span class="ss">f&#39;</span><span class="sc">{</span>ab[(i, j)]<span class="sc">}</span><span class="ss">&#39;</span>, size<span class="op">=</span><span class="dv">20</span>)</span>
<span id="cb7-11"><a href="#cb7-11" aria-hidden="true" tabindex="-1"></a>        axs[i, j].label_outer()</span></code></pre></div>
<pre><code>&lt;&gt;:9: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

&lt;&gt;:9: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

/var/folders/ql/nly59sfj4dd3spkth6lk0kjw0000gn/T/ipykernel_33739/1475610678.py:9: SyntaxWarning:

invalid escape sequence &#39;\D&#39;
</code></pre>
<p><img data-src="analysis_files/figure-revealjs/cell-5-output-2.png" /></p>
</div><div class="column" style="width:45%;">
<ol type="a">
<li><span class="math inline">\(a\Delta t= 2\cdot 1.25=2.5\)</span> and <span class="math inline">\(A=-1.5\)</span>: oscillations and growth</li>
<li><span class="math inline">\(a\Delta t = 2\cdot 0.75=1.5\)</span> and <span class="math inline">\(A=-0.5\)</span>: oscillations and decay</li>
<li><span class="math inline">\(\Delta t=0.5\)</span> and <span class="math inline">\(A=0\)</span>: <span class="math inline">\(u^n=0\)</span> for <span class="math inline">\(n&gt;0\)</span></li>
<li>Smaller <span class="math inline">\(\Delta t\)</span>: qualitatively correct solution</li>
</ol>
</div>
</div>
</section>
<section id="explanation-of-problems-with-crank-nicolson" class="slide level2 smaller">
<h2 class="smaller">Explanation of problems with Crank-Nicolson</h2>
<div class="columns">
<div class="column" style="width:55%;">
<div class="sourceCode" id="cb9"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb9-1"><a href="#cb9-1" aria-hidden="true" tabindex="-1"></a>I, a, T, theta <span class="op">=</span> <span class="dv">1</span>, <span class="dv">2</span>, <span class="dv">8</span>, <span class="fl">0.5</span></span>
<span id="cb9-2"><a href="#cb9-2" aria-hidden="true" tabindex="-1"></a>fig, axs <span class="op">=</span> plt.subplots(<span class="dv">2</span>, <span class="dv">2</span>)</span>
<span id="cb9-3"><a href="#cb9-3" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb9-4"><a href="#cb9-4" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> j <span class="kw">in</span> <span class="bu">range</span>(<span class="dv">2</span>):</span>
<span id="cb9-5"><a href="#cb9-5" aria-hidden="true" tabindex="-1"></a>        u0, t0 <span class="op">=</span> solver(I, a, T, dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j], theta)</span>
<span id="cb9-6"><a href="#cb9-6" aria-hidden="true" tabindex="-1"></a>        axs[i, j].plot(t0, u0, <span class="st">&#39;b&#39;</span>, t0, u_exact(I, a, t0), <span class="st">&#39;r--&#39;</span>)</span>
<span id="cb9-7"><a href="#cb9-7" aria-hidden="true" tabindex="-1"></a>        axs[i, j].legend([<span class="st">&#39;numerical&#39;</span>, <span class="st">&#39;exact&#39;</span>])</span>
<span id="cb9-8"><a href="#cb9-8" aria-hidden="true" tabindex="-1"></a>        axs[i, j].set_title(<span class="ss">f&#39;$\Delta t = </span><span class="sc">{</span>dt[i<span class="op">*</span><span class="dv">2</span><span class="op">+</span>j]<span class="sc">}</span><span class="ss">$&#39;</span>)</span>
<span id="cb9-9"><a href="#cb9-9" aria-hidden="true" tabindex="-1"></a>        axs[i, j].text(<span class="fl">3.2</span>, u0.<span class="bu">max</span>()<span class="op">*</span><span class="fl">0.85</span>, <span class="ss">f&#39;</span><span class="sc">{</span>ab[(i, j)]<span class="sc">}</span><span class="ss">&#39;</span>, size<span class="op">=</span><span class="dv">20</span>)</span>
<span id="cb9-10"><a href="#cb9-10" aria-hidden="true" tabindex="-1"></a>        axs[i, j].label_outer()</span></code></pre></div>
<pre><code>&lt;&gt;:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

&lt;&gt;:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

/var/folders/ql/nly59sfj4dd3spkth6lk0kjw0000gn/T/ipykernel_33739/3300496361.py:8: SyntaxWarning:

invalid escape sequence &#39;\D&#39;
</code></pre>
<p><img data-src="analysis_files/figure-revealjs/cell-6-output-2.png" /></p>
</div><div class="column" style="width:45%;">
<ol type="a">
<li><span class="math inline">\(\Delta t=1.25\)</span> and <span class="math inline">\(A=-0.25\)</span>: oscillatory solution</li>
</ol>
<p>Never any growing solution</p>
</div>
</div>
</section>
<section id="summary-of-stability" class="slide level2 smaller">
<h2 class="smaller">Summary of stability</h2>
<ul>
<li>Forward Euler is <em>conditionally stable</em>
<ul>
<li><span class="math inline">\(\Delta t &lt; 2/a\)</span> for avoiding growth</li>
<li><span class="math inline">\(\Delta t\leq 1/a\)</span> for avoiding oscillations</li>
</ul></li>
<li>The Crank-Nicolson is <em>unconditionally stable</em> wrt growth and conditionally stable wrt oscillations
<ul>
<li><span class="math inline">\(\Delta t &lt; 2/a\)</span> for avoiding oscillations</li>
</ul></li>
<li>Backward Euler is unconditionally stable</li>
</ul>
</section>
<section id="comparing-amplification-factors" class="slide level2 smaller">
<h2 class="smaller">Comparing amplification factors</h2>
<p><span class="math inline">\(u^{n+1}\)</span> is an amplification <span class="math inline">\(A\)</span> of <span class="math inline">\(u^n\)</span>:</p>
<p><span class="math display">\[
u^{n+1} = Au^n,\quad A = \frac{1 - (1-\theta) a\Delta t}{1 + \theta a\Delta t}
\]</span></p>
<p>The exact solution is also an amplification:</p>
<p><span class="math display">\[
\begin{align}
u(t_{n+1}) &amp;= e^{-a(t_n+\Delta t)} \\
u(t_{n+1}) &amp;= e^{-a \Delta t} e^{-a t_n} \\
u(t_{n+1} &amp;= A_e u(t_n), \quad A_e = e^{-a\Delta t}
\end{align}
\]</span></p>
<p>A possible measure of accuracy: <span class="math inline">\(A_e - A\)</span></p>
</section>
<section id="plotting-amplification-factors" class="slide level2">
<h2>Plotting amplification factors</h2>
<div class="sourceCode" id="cb11"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb11-1"><a href="#cb11-1" aria-hidden="true" tabindex="-1"></a>colors <span class="op">=</span> [<span class="st">&#39;g&#39;</span>, <span class="st">&#39;r&#39;</span>, <span class="st">&#39;b&#39;</span>, <span class="st">&#39;m&#39;</span>, <span class="st">&#39;k&#39;</span>]</span>
<span id="cb11-2"><a href="#cb11-2" aria-hidden="true" tabindex="-1"></a>markers <span class="op">=</span> [<span class="st">&#39;o&#39;</span>, <span class="st">&#39;^&#39;</span>, <span class="st">&#39;s&#39;</span>, <span class="st">&#39;&gt;&#39;</span>]</span>
<span id="cb11-3"><a href="#cb11-3" aria-hidden="true" tabindex="-1"></a>N <span class="op">=</span> <span class="dv">10</span></span>
<span id="cb11-4"><a href="#cb11-4" aria-hidden="true" tabindex="-1"></a>adt <span class="op">=</span> np.linspace(<span class="dv">0</span>, <span class="dv">3</span>, N)  </span>
<span id="cb11-5"><a href="#cb11-5" aria-hidden="true" tabindex="-1"></a>An <span class="op">=</span> <span class="kw">lambda</span> t, theta: (<span class="dv">1</span><span class="op">-</span>(<span class="dv">1</span><span class="op">-</span>theta)<span class="op">*</span>t)<span class="op">/</span>(<span class="dv">1</span><span class="op">+</span>theta<span class="op">*</span>t)</span>
<span id="cb11-6"><a href="#cb11-6" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i, th <span class="kw">in</span> <span class="bu">enumerate</span>((<span class="dv">0</span>, <span class="fl">0.5</span>, <span class="dv">1</span>)):</span>
<span id="cb11-7"><a href="#cb11-7" aria-hidden="true" tabindex="-1"></a>  plt.plot(adt, An(adt, th), colors[i], marker<span class="op">=</span>markers[i])</span>
<span id="cb11-8"><a href="#cb11-8" aria-hidden="true" tabindex="-1"></a>plt.plot(adt, np.exp(<span class="op">-</span>adt), colors[<span class="dv">3</span>], marker<span class="op">=</span>markers[<span class="dv">3</span>])</span>
<span id="cb11-9"><a href="#cb11-9" aria-hidden="true" tabindex="-1"></a>plt.legend([<span class="st">&#39;FE&#39;</span>, <span class="st">&#39;CN&#39;</span>, <span class="st">&#39;BE&#39;</span>, <span class="st">&#39;exact&#39;</span>], loc<span class="op">=</span><span class="st">&#39;lower left&#39;</span>)</span>
<span id="cb11-10"><a href="#cb11-10" aria-hidden="true" tabindex="-1"></a>plt.xlabel(<span class="st">&#39;$a\Delta t$&#39;</span>)</span>
<span id="cb11-11"><a href="#cb11-11" aria-hidden="true" tabindex="-1"></a>plt.ylabel(<span class="st">&#39;Amplification factor&#39;</span>)<span class="op">;</span></span></code></pre></div>
<pre><code>&lt;&gt;:10: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

&lt;&gt;:10: SyntaxWarning:

invalid escape sequence &#39;\D&#39;

/var/folders/ql/nly59sfj4dd3spkth6lk0kjw0000gn/T/ipykernel_33739/2338569858.py:10: SyntaxWarning:

invalid escape sequence &#39;\D&#39;
</code></pre>
<p><img data-src="analysis_files/figure-revealjs/cell-7-output-2.png" /></p>
</section>
<section id="padelta-t-is-the-important-parameter-for-numerical-performance" class="slide level2">
<h2><span class="math inline">\(p=a\Delta t\)</span> is the important parameter for numerical performance</h2>
<ul>
<li><span class="math inline">\(p=a\Delta t\)</span> is a dimensionless parameter</li>
<li>all expressions for stability and accuracy involve <span class="math inline">\(p\)</span></li>
<li>Note that <span class="math inline">\(\Delta t\)</span> alone is not so important, it is the combination with <span class="math inline">\(a\)</span> through <span class="math inline">\(p=a\Delta t\)</span> that matters</li>
</ul>
<div class="callout callout-note callout-titled callout-style-default">
<div class="callout-body">
<div class="callout-title">
<div class="callout-icon-container">
<i class='callout-icon'></i>
</div>
<p><strong>Another evidence why <span class="math inline">\(p=a\Delta t\)</span> is key</strong></p>
</div>
<div class="callout-content">
<p>If we scale the model by <span class="math inline">\(\bar t=at\)</span>, <span class="math inline">\(\bar u=u/I\)</span>, we get <span class="math inline">\(d\bar u/d\bar t = -\bar u\)</span>, <span class="math inline">\(\bar u(0)=1\)</span> (no physical parameters!). The analysis show that <span class="math inline">\(\Delta \bar t\)</span> is key, corresponding to <span class="math inline">\(a\Delta t\)</span> in the unscaled model.</p>
</div>
</div>
</div>
</section>
<section id="series-expansion-of-amplification-factors" class="slide level2 smaller">
<h2 class="smaller">Series expansion of amplification factors</h2>
<p>To investigate <span class="math inline">\(A_e - A\)</span> mathematically, we can Taylor expand the expression, using <span class="math inline">\(p=a\Delta t\)</span> as variable.</p>
<div class="sourceCode" id="cb13"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb13-1"><a href="#cb13-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sympy <span class="im">import</span> <span class="op">*</span></span>
<span id="cb13-2"><a href="#cb13-2" aria-hidden="true" tabindex="-1"></a><span class="co"># Create p as a mathematical symbol with name &#39;p&#39;</span></span>
<span id="cb13-3"><a href="#cb13-3" aria-hidden="true" tabindex="-1"></a>p <span class="op">=</span> Symbol(<span class="st">&#39;p&#39;</span>, positive<span class="op">=</span><span class="va">True</span>)</span>
<span id="cb13-4"><a href="#cb13-4" aria-hidden="true" tabindex="-1"></a><span class="co"># Create a mathematical expression with p</span></span>
<span id="cb13-5"><a href="#cb13-5" aria-hidden="true" tabindex="-1"></a>A_e <span class="op">=</span> exp(<span class="op">-</span>p)</span>
<span id="cb13-6"><a href="#cb13-6" aria-hidden="true" tabindex="-1"></a><span class="co"># Find the first 6 terms of the Taylor series of A_e</span></span>
<span id="cb13-7"><a href="#cb13-7" aria-hidden="true" tabindex="-1"></a>A_e.series(p, <span class="dv">0</span>, <span class="dv">6</span>)</span></code></pre></div>
<p><span class="math inline">\(\displaystyle 1 - p + \frac{p^{2}}{2} - \frac{p^{3}}{6} + \frac{p^{4}}{24} - \frac{p^{5}}{120} + O\left(p^{6}\right)\)</span></p>
<p>This is the Taylor expansion of the exact amplification factor. How does it compare with the numerical amplification factors?</p>
</section>
<section id="numerical-amplification-factors" class="slide level2 smaller">
<h2 class="smaller">Numerical amplification factors</h2>
<p>Compute the Taylor expansions of <span class="math inline">\(A_e - A\)</span></p>
<div class="sourceCode" id="cb14"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb14-1"><a href="#cb14-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> IPython.display <span class="im">import</span> display</span>
<span id="cb14-2"><a href="#cb14-2" aria-hidden="true" tabindex="-1"></a>theta <span class="op">=</span> Symbol(<span class="st">&#39;theta&#39;</span>, positive<span class="op">=</span><span class="va">True</span>)</span>
<span id="cb14-3"><a href="#cb14-3" aria-hidden="true" tabindex="-1"></a>A <span class="op">=</span> (<span class="dv">1</span><span class="op">-</span>(<span class="dv">1</span><span class="op">-</span>theta)<span class="op">*</span>p)<span class="op">/</span>(<span class="dv">1</span><span class="op">+</span>theta<span class="op">*</span>p)</span>
<span id="cb14-4"><a href="#cb14-4" aria-hidden="true" tabindex="-1"></a>FE <span class="op">=</span> A_e.series(p, <span class="dv">0</span>, <span class="dv">4</span>) <span class="op">-</span> A.subs(theta, <span class="dv">0</span>).series(p, <span class="dv">0</span>, <span class="dv">4</span>)</span>
<span id="cb14-5"><a href="#cb14-5" aria-hidden="true" tabindex="-1"></a>BE <span class="op">=</span> A_e.series(p, <span class="dv">0</span>, <span class="dv">4</span>) <span class="op">-</span> A.subs(theta, <span class="dv">1</span>).series(p, <span class="dv">0</span>, <span class="dv">4</span>)</span>
<span id="cb14-6"><a href="#cb14-6" aria-hidden="true" tabindex="-1"></a>half <span class="op">=</span> Rational(<span class="dv">1</span>, <span class="dv">2</span>)  <span class="co"># exact fraction 1/2</span></span>
<span id="cb14-7"><a href="#cb14-7" aria-hidden="true" tabindex="-1"></a>CN <span class="op">=</span> A_e.series(p, <span class="dv">0</span>, <span class="dv">4</span>) <span class="op">-</span> A.subs(theta, half).series(p, <span class="dv">0</span>, <span class="dv">4</span>)</span>
<span id="cb14-8"><a href="#cb14-8" aria-hidden="true" tabindex="-1"></a>display(FE)</span>
<span id="cb14-9"><a href="#cb14-9" aria-hidden="true" tabindex="-1"></a>display(BE)</span>
<span id="cb14-10"><a href="#cb14-10" aria-hidden="true" tabindex="-1"></a>display(CN)</span></code></pre></div>
<p><span class="math inline">\(\displaystyle \frac{p^{2}}{2} - \frac{p^{3}}{6} + O\left(p^{4}\right)\)</span></p>
<p><span class="math inline">\(\displaystyle - \frac{p^{2}}{2} + \frac{5 p^{3}}{6} + O\left(p^{4}\right)\)</span></p>
<p><span class="math inline">\(\displaystyle \frac{p^{3}}{12} + O\left(p^{4}\right)\)</span></p>
<ul>
<li>Forward/backward Euler have leading error <span class="math inline">\(p^2\)</span>, or more commonly <span class="math inline">\(\Delta t^2\)</span></li>
<li>Crank-Nicolson has leading error <span class="math inline">\(p^3\)</span>, or <span class="math inline">\(\Delta t^3\)</span></li>
</ul>
</section>
<section id="the-trueglobal-error-at-a-point" class="slide level2">
<h2>The true/global error at a point</h2>
<ul>
<li>The error in <span class="math inline">\(A\)</span> reflects the <strong>local (amplification) error</strong> when going from one time step to the next</li>
<li>What is the <em>global (true) error</em> at <span class="math inline">\(t_n\)</span>?</li>
</ul>
<p><span class="math display">\[
 e^n = u_e(t_n) - u^n = Ie^{-at_n} - IA^n
 \]</span></p>
<ul>
<li>Taylor series expansions of <span class="math inline">\(e^n\)</span> simplify the expression</li>
</ul>
</section>
<section id="computing-the-global-error-at-a-point" class="slide level2 smaller">
<h2 class="smaller">Computing the global error at a point</h2>
<div class="sourceCode" id="cb15"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb15-1"><a href="#cb15-1" aria-hidden="true" tabindex="-1"></a>n <span class="op">=</span> Symbol(<span class="st">&#39;n&#39;</span>, integer<span class="op">=</span><span class="va">True</span>, positive<span class="op">=</span><span class="va">True</span>)</span>
<span id="cb15-2"><a href="#cb15-2" aria-hidden="true" tabindex="-1"></a>u_e <span class="op">=</span> exp(<span class="op">-</span>p<span class="op">*</span>n)   <span class="co"># I=1</span></span>
<span id="cb15-3"><a href="#cb15-3" aria-hidden="true" tabindex="-1"></a>u_n <span class="op">=</span> A<span class="op">**</span>n        <span class="co"># I=1</span></span>
<span id="cb15-4"><a href="#cb15-4" aria-hidden="true" tabindex="-1"></a>FE <span class="op">=</span> u_e.series(p, <span class="dv">0</span>, <span class="dv">4</span>) <span class="op">-</span> u_n.subs(theta, <span class="dv">0</span>).series(p, <span class="dv">0</span>, <span class="dv">4</span>)</span>
<span id="cb15-5"><a href="#cb15-5" aria-hidden="true" tabindex="-1"></a>BE <span class="op">=</span> u_e.series(p, <span class="dv">0</span>, <span class="dv">4</span>) <span class="op">-</span> u_n.subs(theta, <span class="dv">1</span>).series(p, <span class="dv">0</span>, <span class="dv">4</span>)</span>
<span id="cb15-6"><a href="#cb15-6" aria-hidden="true" tabindex="-1"></a>CN <span class="op">=</span> u_e.series(p, <span class="dv">0</span>, <span class="dv">4</span>) <span class="op">-</span> u_n.subs(theta, half).series(p, <span class="dv">0</span>, <span class="dv">4</span>)</span>
<span id="cb15-7"><a href="#cb15-7" aria-hidden="true" tabindex="-1"></a>display(simplify(FE))</span>
<span id="cb15-8"><a href="#cb15-8" aria-hidden="true" tabindex="-1"></a>display(simplify(BE))</span>
<span id="cb15-9"><a href="#cb15-9" aria-hidden="true" tabindex="-1"></a>display(simplify(CN))</span></code></pre></div>
<p><span class="math inline">\(\displaystyle \frac{n p^{2}}{2} + \frac{n p^{3}}{3} - \frac{n^{2} p^{3}}{2} + O\left(p^{4}\right)\)</span></p>
<p><span class="math inline">\(\displaystyle - \frac{n p^{2}}{2} + \frac{n p^{3}}{3} + \frac{n^{2} p^{3}}{2} + O\left(p^{4}\right)\)</span></p>
<p><span class="math inline">\(\displaystyle \frac{n p^{3}}{12} + O\left(p^{4}\right)\)</span></p>
<p>Substitute <span class="math inline">\(n\)</span> by <span class="math inline">\(t/\Delta t\)</span> and <span class="math inline">\(p\)</span> by <span class="math inline">\(a \Delta t\)</span>:</p>
<ul>
<li>Forward and Backward Euler: leading order term <span class="math inline">\(\scriptstyle\frac{1}{2}ta^2\Delta t\)</span></li>
<li>Crank-Nicolson: leading order term <span class="math inline">\(\frac{1}{12}ta^3\Delta t^2\)</span></li>
</ul>
</section>
<section id="convergence" class="slide level2">
<h2>Convergence</h2>
<p>The numerical scheme is convergent if the global error <span class="math inline">\(e^n\rightarrow 0\)</span> as <span class="math inline">\(\Delta t\rightarrow 0\)</span>. If the error has a leading order term <span class="math inline">\((\Delta t)^r\)</span>, the convergence rate is of order <span class="math inline">\(r\)</span>.</p>
</section>
<section id="integrated-errors" class="slide level2 smaller">
<h2 class="smaller">Integrated errors</h2>
<p>The <span class="math inline">\(\ell^2\)</span> norm of the numerical error is computed as</p>
<p><span class="math display">\[
||e^n||_{\ell^2} = \sqrt{\Delta t\sum_{n=0}^{N_t} ({u_{e}}(t_n) - u^n)^2}
\]</span></p>
<p>We can compute this using Sympy. Forward/Backward Euler has <span class="math inline">\(e^n \sim np^2/2\)</span></p>
<div class="sourceCode" id="cb16"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb16-1"><a href="#cb16-1" aria-hidden="true" tabindex="-1"></a>h, N, a, T <span class="op">=</span> symbols(<span class="st">&#39;h,N,a,T&#39;</span>) <span class="co"># h represents Delta t</span></span>
<span id="cb16-2"><a href="#cb16-2" aria-hidden="true" tabindex="-1"></a>simplify(sqrt(h <span class="op">*</span> summation((n<span class="op">*</span>p<span class="op">**</span><span class="dv">2</span><span class="op">/</span><span class="dv">2</span>)<span class="op">**</span><span class="dv">2</span>, (n, <span class="dv">0</span>, N))).subs(p, a<span class="op">*</span>h).subs(N, T<span class="op">/</span>h))</span></code></pre></div>
<p><span class="math inline">\(\displaystyle \frac{\sqrt{6} a^{2} h^{2} \sqrt{T \left(\frac{2 T^{2}}{h^{2}} + \frac{3 T}{h} + 1\right)}}{12}\)</span></p>
<p>If we keep only the leading term in the parenthesis, we get the first order <span class="math display">\[
||e^n||_{\ell^2} \approx \frac{1}{2}\sqrt{\frac{T^3}{3}} a^2\Delta t
\]</span></p>
</section>
<section id="crank-nicolson" class="slide level2 smaller">
<h2 class="smaller">Crank-Nicolson</h2>
<p>For Crank-Nicolson the pointwise error is <span class="math inline">\(e^n \sim n p^3 / 12\)</span>. We get</p>
<div class="sourceCode" id="cb17"><pre class="sourceCode python cell-code"><code class="sourceCode python"><span id="cb17-1"><a href="#cb17-1" aria-hidden="true" tabindex="-1"></a>simplify(sqrt(h <span class="op">*</span> summation((n<span class="op">*</span>p<span class="op">**</span><span class="dv">3</span><span class="op">/</span><span class="dv">12</span>)<span class="op">**</span><span class="dv">2</span>, (n, <span class="dv">0</span>, N))).subs(p, a<span class="op">*</span>h).subs(N, T<span class="op">/</span>h))</span></code></pre></div>
<p><span class="math inline">\(\displaystyle \frac{\sqrt{6} a^{3} h^{3} \sqrt{T \left(\frac{2 T^{2}}{h^{2}} + \frac{3 T}{h} + 1\right)}}{72}\)</span></p>
<p>which is simplified to the second order accurate</p>
<p><span class="math display">\[
||e^n||_{\ell^2} \approx \frac{1}{12}\sqrt{\frac{T^3}{3}}a^3\Delta t^2
\]</span></p>
<div class="callout callout-note callout-titled callout-style-default">
<div class="callout-body">
<div class="callout-title">
<div class="callout-icon-container">
<i class='callout-icon'></i>
</div>
<p><strong>Summary of errors</strong></p>
</div>
<div class="callout-content">
<p>Analysis of both the pointwise and the time-integrated true errors:</p>
<ul>
<li>1st order for Forward and Backward Euler</li>
<li>2nd order for Crank-Nicolson</li>
</ul>
</div>
</div>
</div>
</section>
<section id="truncation-error" class="slide level2 smaller">
<h2 class="smaller">Truncation error</h2>
<ul>
<li>How good is the discrete equation?</li>
<li>Possible answer: see how well <span class="math inline">\(u_{e}\)</span> fits the discrete equation</li>
</ul>
<p>Consider the forward difference equation <span class="math display">\[ 
\frac{u^{n+1}-u^n}{\Delta t} = -au^n
\]</span></p>
<p>Insert <span class="math inline">\(u_{e}\)</span> to obtain a truncation error <span class="math inline">\(R^n\)</span></p>
<p><span class="math display">\[
\frac{u_{e}(t_{n+1})-u_{e}(t_n)}{\Delta t} + au_{e}(t_n) = R^n \neq 0
\]</span></p>
</section>
<section id="computation-of-the-truncation-error" class="slide level2 smaller">
<h2 class="smaller">Computation of the truncation error</h2>
<ul>
<li>The residual <span class="math inline">\(R^n\)</span> is the <strong>truncation error</strong>. How does <span class="math inline">\(R^n\)</span> vary with <span class="math inline">\(\Delta t\)</span>?</li>
</ul>
<p>Tool: Taylor expand <span class="math inline">\(u_{e}\)</span> around the point where the ODE is sampled (here <span class="math inline">\(t_n\)</span>)</p>
<p><span class="math display">\[ 
u_{e}(t_{n+1}) = u_{e}(t_n) + u_{e}&#39;(t_n)\Delta t + \frac{1}{2}u_{e}&#39;&#39;(t_n)
\Delta t^2 + \cdots
\]</span></p>
<p>Inserting this Taylor series for <span class="math inline">\(u_{e}\)</span> in the forward difference equation</p>
<p><span class="math display">\[
R^n = \frac{u_{e}(t_{n+1})-u_{e}(t_n)}{\Delta t} + au_{e}(t_n) 
\]</span></p>
<p>to get</p>
<p><span class="math display">\[ 
R^n = u_{e}&#39;(t_n) + \frac{1}{2}u_{e}&#39;&#39;(t_n)\Delta t + \ldots + au_{e}(t_n)
\]</span></p>
</section>
<section id="the-truncation-error-forward-euler" class="slide level2 smaller">
<h2 class="smaller">The truncation error forward Euler</h2>
<p>We have <span class="math display">\[ 
R^n = u_{e}&#39;(t_n) + \frac{1}{2}u_{e}&#39;&#39;(t_n)\Delta t + \ldots + au_{e}(t_n)
\]</span></p>
<p>Since <span class="math inline">\(u_{e}\)</span> solves the ODE <span class="math inline">\(u_{e}&#39;(t_n)=-au_{e}(t_n)\)</span>, we get that <span class="math inline">\(u_{e}&#39;(t_n)\)</span> and <span class="math inline">\(au_{e}(t_n)\)</span> cancel out. We are left with leading term</p>
<p><span class="math display">\[ 
R^n \approx \frac{1}{2}u_{e}&#39;&#39;(t_n)\Delta t
\]</span></p>
<p>This is a mathematical expression for the truncation error.</p>
</section>
<section id="the-truncation-error-for-other-schemes" class="slide level2 smaller">
<h2 class="smaller">The truncation error for other schemes</h2>
<p>Backward Euler:</p>
<p><span class="math display">\[ 
R^n \approx -\frac{1}{2}u_{e}&#39;&#39;(t_n)\Delta t 
\]</span></p>
<p>Crank-Nicolson:</p>
<p><span class="math display">\[
R^{n+\scriptstyle\frac{1}{2}} \approx \frac{1}{24}u_{e}&#39;&#39;&#39;(t_{n+\scriptstyle\frac{1}{2}})\Delta t^2
\]</span></p>
</section>
<section id="consistency-stability-and-convergence" class="slide level2 smaller">
<h2 class="smaller">Consistency, stability, and convergence</h2>
<ul>
<li><p><em>Truncation error</em> measures the residual in the difference equations. The scheme is <em>consistent</em> if the truncation error goes to 0 as <span class="math inline">\(\Delta t\rightarrow 0\)</span>. Importance: the difference equations approaches the differential equation as <span class="math inline">\(\Delta t\rightarrow 0\)</span>.</p></li>
<li><p><em>Stability</em> means that the numerical solution exhibits the same qualitative properties as the exact solution. Here: monotone, decaying function.</p></li>
<li><p><em>Convergence</em> implies that the true (global) error <span class="math inline">\(e^n =u_{e}(t_n)-u^n\rightarrow 0\)</span> as <span class="math inline">\(\Delta t\rightarrow 0\)</span>. This is really what we want!</p></li>
</ul>
<p>The Lax equivalence theorem for <em>linear</em> differential equations: consistency + stability is equivalent with convergence.</p>
<p>(Consistency and stability is in most problems much easier to establish than convergence.)</p>
</section>
