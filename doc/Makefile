SHELL:=/bin/bash -O extglob
NAME=test
ARCHIVEDIR=./${NAME}_tex

archive: archive_utf8 archive_cp1251

pdf:
	pdflatex --jobname ${NAME} major.tex
	bibtex ${NAME}
	pdflatex --jobname ${NAME} major.tex
	pdflatex --jobname ${NAME} major.tex
ps:
	latex --jobname ${NAME} major.tex
	bibtex ${NAME}
	latex --jobname ${NAME} major.tex
	latex --jobname ${NAME} major.tex
	dvips ${NAME}.dvi
archive_utf8:
	mkdir -p ${ARCHIVEDIR}
	cp -r pics ${ARCHIVEDIR}
	cp *.tex *.bib *.sty ${ARCHIVEDIR}
	zip -r  ${ARCHIVEDIR}.zip ${ARCHIVEDIR}
archive_cp1251:
	mkdir -p ${ARCHIVEDIR}
	cp -r pics ${ARCHIVEDIR}
	cp *.tex *.bib *.sty ${ARCHIVEDIR}
	sed -i -e 's|\[utf8\]|\[cp1251\]|g' ${ARCHIVEDIR}/major.tex
	enconv -L ru -x cp1251 ${ARCHIVEDIR}/*.tex
	zip -r ${ARCHIVEDIR}_cp1251.zip ${ARCHIVEDIR}
clean:
	rm -f *.aux *.log *.*~ *.bbl *.blg *.out *.dvi
