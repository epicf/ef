SHELL:=/bin/bash -O extglob
NAME=epicf
ARCHIVEDIR=./${NAME}_tex
MAJOR=epicf.tex

pdf:
	pdflatex --jobname ${NAME} ${MAJOR}
	-bibtexu ${NAME}
	pdflatex --jobname ${NAME} ${MAJOR}
	pdflatex --jobname ${NAME} ${MAJOR}

ps:
	latex --jobname ${NAME} ${MAJOR}
	bibtexu ${NAME}
	latex --jobname ${NAME} ${MAJOR}
	latex --jobname ${NAME} ${MAJOR}
	dvips ${NAME}.dvi

archive_utf8:
	mkdir -p ${ARCHIVEDIR}
	cp -r figs ${ARCHIVEDIR}
	cp *.tex *.bib *.sty ${ARCHIVEDIR}
	zip -r  ${ARCHIVEDIR}.zip ${ARCHIVEDIR}

archive_cp1251:
	mkdir -p ${ARCHIVEDIR}
	cp -r figs ${ARCHIVEDIR}
	cp *.sty ${ARCHIVEDIR}
	for	x in *.tex; do iconv -c -f utf8 -t cp1251 -o ${ARCHIVEDIR}/$${x} $${x}; done
	for	x in *.bib; do iconv -c -f utf8 -t cp1251 -o ${ARCHIVEDIR}/$${x} $${x}; done
	sed -i -e 's|\[utf8\]|\[cp1251\]|g' ${ARCHIVEDIR}/${MAJOR}
	zip -r ${ARCHIVEDIR}_cp1251.zip ${ARCHIVEDIR}

archive: archive_utf8 archive_cp1251

all: pdf archive

clean:
	rm -f *.aux *.log *.*~ *.bbl *.blg *.out *.dvi
