%& -no-cctspace
% LaTeX class for Chinese Control Conference papers
%
% $ based on Id: cccconf.cls,v 1.5 2007/03/08 07:29:46 zlb Exp $
% $Id: cccconf.cls 4 2011-01-14 11:04:47Z hsqi $

% 2010/02/07: make thebibliography environment be compatible with natbib package
% 2010/01/13: comment out ``\renewcommand\newblock{\par}'' (reported by mathmhb)

\newif\ifUseMulticol
\UseMulticolfalse % change to true to use 'multicol' package.

%
% Note: when using the 'multicol' package, the 'figure' and 'table'
% environments are modified to ignore user provided placement specifiers
% and always use the '[H]' specifier of the 'float' package.
%

\ProvidesClass{cccconf}[2010/02/10]
\NeedsTeXFormat{LaTeX2e}[1994/12/01]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\newif\ifChinese
\Chinesetrue
\DeclareOption{english}{\Chinesefalse}
\DeclareOption{chinese}{\Chinesetrue}
\DeclareOption{usemulticol}{\UseMulticoltrue}
\ProcessOptions

\ifUseMulticol
  \LoadClass[a4paper]{article}[1996/10/31]
  \RequirePackage{multicol}
  \RequirePackage{float}
  \AtEndDocument{\end{multicols}}
  % Force 'figure' and 'table' to use [H] placement
  \let\figure@bak\figure
  \renewcommand\figure[1][]{\figure@bak[H]}
  \let\table@bak\table
  \renewcommand\table[1][]{\table@bak[H]}
  \let\balance\relax
  \long\def\singlecolumn#1{\end{multicols}#1\begin{multicols}{2}}
\else
  \LoadClass[a4paper,twocolumn]{article}[1996/10/31]
  \RequirePackage{balance}
  \long\def\singlecolumn#1{\errmessage{You must add 'usemulticol' option in
    order to use \string\singlecolumn\space command}}
\fi
\pagestyle{empty}
\RequirePackage{times}

\def\SetEnglishNames{%
  \def\emailname{E-mail}%
  \def\abstractname{Abstract}%
  \def\keywordsname{Key Words}
  \def\arefsuffix{@eng}}

\newif\ifsecondtitle
\ifChinese
  \RequirePackage[CJK]{cctbase}[2003/01/10]
%  \RequirePackage[CJKbookmarks]{hyperref}
  \def\emailname{E-mail}
  \def\abstractname{摘^^a1^^a1要}
  \def\keywordsname{关键词}
  \def\tablename{表}
  \def\figurename{图}
  \def\refname{参考文献}
  \def\arefsuffix{@chn}
  \let\biaosong\heiti
  \secondtitletrue
\else
%  \RequirePackage{hyperref}
  \def\figurename{Fig.}
  \SetEnglishNames
  \secondtitlefalse
\fi

%\newcommand\URL[1]{\href{#1}{{#1}}}

%---------------------------------- Page layout
\RequirePackage{graphicx}
\RequirePackage[left=1.7cm,right=1.7cm,top=2cm,bottom=2.2cm]{geometry}
\RequirePackage{indentfirst}


\ifChinese\renewcommand\baselinestretch{1.1}\fi
\columnsep=0.7cm
\parskip=\z@

\tolerance=5000
\hbadness=5000
\vbadness=5000

\ifChinese
  %\let\small@bak\small
  %\def\small{\small@bak\zihao{6}}

  %\let\normalsize@bak\normalsize
  %\def\normalsize{\normalsize@bak\zihao{-5}}

  \let\footnotesize@bak\footnotesize
  \def\footnotesize{\footnotesize@bak\zihao{6}}
  \AtBeginDocument{\parindent=2\ccwd}
\else
  \AtBeginDocument{\parindent=1em}
\fi

%---------------------------------- Title, abstract, and key words

\newbox\titleb@x
\setbox\titleb@x\vbox{}

\newcounter{affno}
\setcounter{affno}{0}

\def\aref#1{\let\@aref@comma\relax{\small$^{\@aref#1,,}$}}

\def\@aref#1,{\def\@tempa{#1}\ifx\@tempa\@empty \let\@next\relax\else
  \@aref@comma\ref{#1\arefsuffix}\let\@next\@aref\fi\let\@aref@comma,\@next}

\newtoks\@afflist
\@afflist={}
\newcommand\affiliation[2][]{\@afflist=\expandafter{\the\@afflist \par
  \fontsize{9}{10}\selectfont \def\@tempa{#1}\ifx\@tempa\@empty\else
    \refstepcounter{affno}\label{#1\arefsuffix}\arabic{affno}. \fi
  #2\par\smallskip}}

\let\thefootnote@bak\thefootnote	% save \the footnote
\renewcommand\thefootnote{}
\def\maketitle{\def\saved@title{\let \footnote \thanks \parskip\z@
  \bgroup
  \begin{center}
    {\Large\bfseries
     \ifChinese
      \vspace*{-0.7cm}%
      \heiti \zihaoAny{18}%
      \ifsecondtitle \else
        \vspace*{1.6cm}\fontsize{16}{20}\selectfont %16,20 % 14,16
      \fi
     \else
      \vspace*{-0.8cm}%\vspace*{-0.2cm}
      \fontsize{18}{22}\selectfont   %16,20
     \fi
     \@title\par}\bigskip
    %{\fontsize{11}{12}\selectfont\ifChinese\large\fangsong\fi \@author}\\ %normalsize=>11pt
    {\ifChinese\fangsong\fi \@author}\\ %normalsize=>11pt
    \smallskip
    \the\@afflist 
  \end{center}%
  \egroup
  \par}}%bigskip

\newbox\@abstract
\renewenvironment{abstract}{\global\setbox\@abstract\vbox\bgroup
  %\let\thefootnote\thefootnote@bak
  \hsize=\textwidth\leftskip=0.7cm \rightskip=0.7cm \noindent
  \small \textbf{\abstractname: }%
  \def\@tempa{Abstract}\ifx\@tempa\abstractname 
  %\renewcommand\baselinestretch{3.0}
 \baselineskip=10pt
 % AAA 
  \fi
 }{\par\egroup} % save \the abstract

\newif\ifhavekeywords \havekeywordsfalse
\def\keywords#1{
  \setbox\titleb@x\vbox{\hsize=\textwidth \box\titleb@x
  \def\email##1{\unskip\\\emailname: {##1}}%
  \saved@title\box\@abstract\par\medskip
  \def\@tempa{#1}\ifx\@tempa\@empty\else
    \vbox{\leftskip=0.7cm \rightskip=0.7cm \small
      \noindent\textbf{\keywordsname: }#1}\par
  \fi
  \ifsecondtitle \vspace*{-1.5cm}%\vspace*{-0.5cm}
  \fi
  \bigskip}%
  \@afflist={}\setcounter{affno}{0}%
  \ifsecondtitle
    \SetEnglishNames
    \secondtitlefalse
  \else
    \global\havekeywordstrue
    \secondtitletrue
    \ifUseMulticol
        \vbox{\vskip 1.25cm}%
        \box\titleb@x\par\vskip 0.3cm % 0.8cm
      \begin{multicols}{2}
      \let\@footnotetext\orig@footnotetext
    \else
      \twocolumn[%
        \vskip 1.25cm %
        \box\titleb@x\par\vskip 0.3cm % 0.8cm
      ]%
    \fi
    \setbox\titleb@x\vbox{}%
  \fi
}

%---------------------------------- Sections

\renewcommand\section{\ifChinese\ifsecondtitle\else
		      \ClassError{cccconf}
			{Missing \string\keywords\space or English title}
			{English title/abstract/keywords are required}%
		      \maketitle\keywords{}%
		      \fi\fi
		      \ifhavekeywords\else\keywords{}\fi
		      \ifx\thefootnote\thefootnote@bak\else
		        \setcounter{footnote}{0}\let\thefootnote\thefootnote@bak
		      \fi
		      \@startsection{section}{1}{\z@}%
                                    {-6pt \@plus -1ex \@minus -1ex}%
                                    {6pt \@plus .2ex}%
                                    {\normalfont\fontsize{11}{12}\bfseries}}%11,12 %12,14
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%,
                                       {-6pt }% \@plus -1ex \@minus -1ex}%
                                       {3pt \@plus .0ex}%
                                       {\normalfont\normalsize\bfseries}}
\ifChinese
  % Restore section definition
  \def\@seccntformat#1{\csname the#1\endcsname\hskip 1em}
  \def\@sect#1#2#3#4#5#6[#7]#8{%
    \ifnum #2>\c@secnumdepth
      \let\@svsec\@empty
    \else
      \refstepcounter{#1}%
      \protected@edef\@svsec{\@seccntformat{#1}\relax}%
    \fi
    \@tempskipa #5\relax
    \ifdim \@tempskipa>\z@
      \begingroup
        #6{%
          \@hangfrom{\hskip #3\relax\@svsec}%
            \interlinepenalty \@M #8\@@par}%
      \endgroup
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #7}%
    \else
      \def\@svsechd{%
        #6{\hskip #3\relax
        \@svsec #8}%
        \csname #1mark\endcsname{#7}%
        \addcontentsline{toc}{#1}{%
          \ifnum #2>\c@secnumdepth \else
            \protect\numberline{\csname the#1\endcsname}%
          \fi
          #7}}%
    \fi
    \@xsect{#5}}
  %\renewcommand\thesection{\arabic{section}}
  %\renewcommand\thesubsection{\thesection.\arabic{subsection}}
  %\renewcommand\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\else
  \renewcommand\refname{References}
\fi

%---------------------------------- Top/bottom rules in tabulars
%%%\newcommand\hhline{%
%%%  \noalign{\ifnum0=`}\fi\hrule \@height 2\arrayrulewidth \futurelet
%%%   \@tempa\@xhline}
\newcommand\hhline{\hline\noalign{\vskip -2pt}\hline\noalign{\vskip 1pt}}

%---------------------------------- Floats
% Note: the values below are taken from ieeeconf.cls

\floatsep 1\baselineskip plus  0.2\baselineskip minus  0.2\baselineskip
\textfloatsep 1.7\baselineskip plus  0.2\baselineskip minus  0.4\baselineskip
\@fptop 0pt plus 1fil
\@fpsep 0.75\baselineskip plus 2fil
\@fpbot 0pt plus 1fil
\def\topfraction{1.0}
\def\bottomfraction{.4}
\def\floatpagefraction{0.8}
\def\textfraction{.2}

\dblfloatsep 1\baselineskip plus  0.2\baselineskip minus  0.2\baselineskip
\dbltextfloatsep 1.7\baselineskip plus  0.2\baselineskip minus  0.4\baselineskip

\@dblfptop 0pt plus 1fil
\@dblfpsep 0.75\baselineskip plus 2fil
\@dblfpbot 0pt plus 1fil
\def\dbltopfraction{1.0}
\def\dblfloatpagefraction{0.8}
\setcounter{dbltopnumber}{4}

\intextsep 1\baselineskip plus 0.2\baselineskip minus  0.2\baselineskip
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}

\setlength\abovecaptionskip{0.25\baselineskip}
\setlength\belowcaptionskip{0.25\baselineskip}

%---------------------------------- List spacing controls
% Note: the values below are taken from IEEEtran.cls
\topsep           0.5\baselineskip
\partopsep          \z@
\parsep             \z@
\itemsep            \z@
\itemindent         -1em
\leftmargin         2em
\leftmargini        2em
%\itemindent         2em  % Alternative values: sometimes used.
%\leftmargini        0em
\leftmarginii       1em
\leftmarginiii    1.5em
\leftmarginiv     1.5em
\leftmarginv      1.0em
\leftmarginvi     1.0em
\labelsep         0.5em 
\labelwidth         \z@
\def\@listi{\leftmargin\leftmargini \topsep 2pt plus 1pt minus 1pt}
\let\@listI\@listi
\def\@listii{\leftmargin\leftmarginii\labelwidth\leftmarginii%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listiii{\leftmargin\leftmarginiii\labelwidth\leftmarginiii%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listiv{\leftmargin\leftmarginiv\labelwidth\leftmarginiv%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listv{\leftmargin\leftmarginv\labelwidth\leftmarginv%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listvi{\leftmargin\leftmarginvi\labelwidth\leftmarginvi%
    \advance\labelwidth-\labelsep \topsep 2pt}

\def\labelenumi{\theenumi)}     \def\theenumi{\arabic{enumi}}
\def\labelenumii{\theenumii)}  \def\theenumii{\alph{enumii}}
\def\labelenumiii{\theenumiii)} \def\theenumiii{\roman{enumiii}}
\def\labelenumiv{\theenumiv)}   \def\theenumiv{\Alph{enumiv}}
\def\p@enumii{\theenumi}
\def\p@enumiii{\theenumi(\theenumii)}
\def\p@enumiv{\p@enumiii\theenumiii}

\def\labelitemi{$\scriptstyle\bullet$}
\def\labelitemii{\textbf{--}}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}

%---------------------------------- The bibliography

\renewcommand\@openbib@code{\parsep \z@ \itemsep \z@ \parskip \z@
  \small\ifChinese\zihao{-5}\fi}
% \renewcommand\newblock{\par}
\AtBeginDocument{
\ifcsname bibsep\endcsname\setlength{\bibsep}{\z@}\fi
\ifcsname bibfont\endcsname\ifChinese\def\bibfont{\zihao{-5}}\else\def\bibfont{\small}\fi\fi
}

%---------------------------------- Floats

\def \@floatboxreset {%
  \reset@font
  \small
  \ifChinese\renewcommand\arraystretch{1.2}\fi
  \@setminipage
}

\endinput
