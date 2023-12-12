      logical function gread(unit1,init2,echo,object)
      integer unit1,unit2,quote,iary(*)
      real rary(*)
      real*8 d,dary(*)
c      logical object,echo,dblank,image/.false./,new
      logical object,echo,dblank,image,new
      logical aread,bread,cread,dread,fread,iread,sread,tread
      character*(*) name,sary
      character*80 card
c      character*4 eof/'eof '/,blanks/'    '/
      character*4 eof,blanks
c      character*1 blank/' '/,comma/','/
      character*1 blank,comma
c
      save unit2,card,length,ieq,quote,image
c
      image=.false.
      eof='eof'
      blanks='    '
      blank=' '
      comma=','
c
      unit2=init2
      if(image.and.object) write(unit2,102)
      write(unit2,'('' >'',$)')
      read(unit1,100,end=2) card
      if(echo) write(unit2,101) card
      image=dblank(card,80,length,quote)
      if(length.eq.0) go to 2
      ieq=index(card,'=')
      return
2     card(1:4)=eof
      write(unit2,103) eof
      length=3
      image=.true.
      quote=0
      ieq=0
      return
c
      entry bread(rary,lf,lmax,new)
      gread=.false.
      if(ieq.ne.0.or.quote.ne.0.or..not.image) return
      int=2
      go to 8
c
      entry iread(name,iary,lf,lmax,new)
      int=1
      go to 7
c
      entry fread(name,rary,lf,lmax,new)
      int=2
      go to 7
c
      entry dread(name,dary,lf,lmax,new)
      int=3
c
7     l=len(name)
      gread=.false.
      if(ieq-1.ne.l.or..not.image) return
      if(name(:l).ne.card(:l)) return
      if(quote.ne.0) then
         write(unit2,104)
         image=.false.
         return
      endif
8     new=.true.
      gread=.true.
      call getdp(d,card,ieq,length,.false.)
      do 6 i=1,lmax
      if(int.eq.2) then
         d=rary(i)
      else if(int.eq.1) then
         d=iary(i)
      else if(int.eq.3) then
         d=dary(i)
      endif
      call getdp(d,card,ieq,length,image)
      if(int.eq.2) then
         rary(i)=d
      else if(int.eq.1) then
         iary(i)=d
      else if(int.eq.3) then
         dary(i)=d
      endif
      lf=i
      if(length.eq.0) write(unit2,104)
      if(.not.image) return
6     continue
      lf=lmax
      image=.false.
      return
c
      entry cread(name)
      l=len(name)
      gread=.false.
      if(ieq.ne.0.or.length.ne.l.or..not.image) return
      image=name(:l).ne.card(:l)
      gread=.not.image
      if(.not.(gread.and.quote.ne.0)) return
      write(unit2,104)
      gread=.false.
      return
c
      entry sread(name,sary,new)
      l=len(name)
      gread=.false.
      if(ieq-1.ne.l.or..not.image) return
      if(name(:l).ne.card(:l)) return
      image=.false.
      if(quote.ne.ieq+1) then
         write(unit2,104)
         return
      endif
3     new=.true.
      gread=.true.
      sary=card(quote:length)
      return
c
      entry tread(sary,new)
      gread=.false.
      if(ieq.ne.0.or.quote.eq.0.or..not.image) return
      image=.false.
      go to 3
c
100   format(a80)
101   format(1x,a80)
102   format(1x,'??????????')
103   format(1x,a4)
104   format(1x,'data format error.')
      end